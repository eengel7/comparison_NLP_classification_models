import logging
import math
import os
import warnings
from dataclasses import asdict

import pandas as pd
import torch

from torch.optim import AdamW
from torch.utils.data import (DataLoader, RandomSampler)
from src.utils import calculate_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    Adafactor, get_constant_schedule, get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)

from src.classification.classification_utils import (InputExample,
                                                     flatten_results,
                                                     load_hf_dataset)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


def train_model(
        classification_model,
        train_df,
        multi_label=True,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, classification_model.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the classification_model. Any changes made will persist for the classification_model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            classification_model.args.update_from_dict(args)

        if classification_model.args.silent:
            show_running_loss = False

        if classification_model.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to classification_model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = f"{classification_model.args.output_dir}{classification_model.args.model_type}/"

        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not classification_model.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )
        classification_model._move_model_to_device()

        if classification_model.args.use_hf_datasets:
            if classification_model.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if classification_model.args.model_type in ["layoutlm", "layoutlmv2"]:
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            train_dataset = load_hf_dataset(
                train_df, classification_model.tokenizer, classification_model.args, multi_label=multi_label
            )
        else:
            if "text" in train_df.columns and "labels" in train_df.columns:

                train_examples = (
                    train_df["text"].astype(str).tolist(),
                    train_df["labels"].tolist(),
                )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                train_examples = (
                    train_df.iloc[:, 0].astype(str).tolist(),
                    train_df.iloc[:, 1].tolist(),
                )
            train_dataset = classification_model.load_and_cache_examples(
                train_examples, verbose=verbose
            )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=classification_model.args.train_batch_size,
            num_workers=classification_model.args.dataloader_num_workers,
        )

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = train(classification_model,
            train_dataloader,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        # model_to_save = classification_model.classification_model.module if hasattr(classification_model.model, "module") else classification_model.model
        # model_to_save.save_pretrained(output_dir)
        # classification_model.tokenizer.save_pretrained(output_dir)
        # torch.save(classification_model.args, os.path.join(output_dir, "training_args.bin"))
        classification_model.save_model(model=classification_model.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    classification_model.args.model_type, output_dir
                )
            )

        return global_step, training_details

def train(
    classification_model,
    train_dataloader,
    output_dir,
    multi_label=True,
    show_running_loss=True,
    eval_df=None,
    test_df=None,
    verbose=True,
    **kwargs,
):
    """
    Trains the model on train_dataset.

    Utility function to be used by the train_model() method. Not intended to be used directly.
    """

    model = classification_model.model
    args = classification_model.args

    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = []
    custom_parameter_names = set()
    for group in classification_model.args.custom_parameter_groups:
        params = group.pop("params")
        custom_parameter_names.update(params)
        param_group = {**group}
        param_group["params"] = [
            p for n, p in model.named_parameters() if n in params
        ]
        optimizer_grouped_parameters.append(param_group)

    for group in classification_model.args.custom_layer_parameters:
        layer_number = group.pop("layer")
        layer = f"layer.{layer_number}."
        group_d = {**group}
        group_nd = {**group}
        group_nd["weight_decay"] = 0.0
        params_d = []
        params_nd = []
        for n, p in model.named_parameters():
            if n not in custom_parameter_names and layer in n:
                if any(nd in n for nd in no_decay):
                    params_nd.append(p)
                else:
                    params_d.append(p)
                custom_parameter_names.add(n)
        group_d["params"] = params_d
        group_nd["params"] = params_nd

        optimizer_grouped_parameters.append(group_d)
        optimizer_grouped_parameters.append(group_nd)

    if not classification_model.args.train_custom_parameters_only:
        optimizer_grouped_parameters.extend(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n not in custom_parameter_names
                        and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n not in custom_parameter_names
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        )

    warmup_steps = math.ceil(t_total * args.warmup_ratio)
    args.warmup_steps = (
        warmup_steps if args.warmup_steps == 0 else args.warmup_steps
    )

    if args.optimizer == "AdamW":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=args.adam_betas,
        )
    elif args.optimizer == "Adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adafactor_eps,
            clip_threshold=args.adafactor_clip_threshold,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.adafactor_beta1,
            weight_decay=args.weight_decay,
            scale_parameter=args.adafactor_scale_parameter,
            relative_step=args.adafactor_relative_step,
            warmup_init=args.adafactor_warmup_init,
        )

    else:
        raise ValueError(
            "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                args.optimizer
            )
        )

    if args.scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)

    elif args.scheduler == "constant_schedule_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    elif args.scheduler == "linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
        )

    elif args.scheduler == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            num_cycles=args.cosine_schedule_num_cycles,
        )

    elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            num_cycles=args.cosine_schedule_num_cycles,
        )

    elif args.scheduler == "polynomial_decay_schedule_with_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            lr_end=args.polynomial_decay_schedule_lr_end,
            power=args.polynomial_decay_schedule_power,
        )

    else:
        raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    training_progress_scores = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
    )
    epoch_number = 0
    best_eval_metric = None
    early_stopping_counter = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = 0
    current_loss = "Initializing"

    if args.model_name and os.path.exists(args.model_name):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name.split("/")[-1].split("-")
            if len(checkpoint_suffix) > 2:
                checkpoint_suffix = checkpoint_suffix[1]
            else:
                checkpoint_suffix = checkpoint_suffix[-1]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "   Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("   Continuing training from epoch %d", epochs_trained)
            logger.info("   Continuing training from global step %d", global_step)
            logger.info(
                "   Will skip the first %d steps in the current epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("   Starting fine-tuning.")

    if args.evaluate_during_training:
        training_progress_scores = classification_model._create_training_progress_scores(
            multi_label, **kwargs
        )

    if args.wandb_project:
        if not wandb.setup().settings.sweep_id:
            logger.info(" Initializing WandB run for training.")
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="test_code_structure")
            classification_model.wandb_run_id = wandb.run.id
        wandb.watch(classification_model.model)

    for _ in train_iterator:
        model.train()
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        train_iterator.set_description(
            f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
        )
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
            disable=args.silent,
            mininterval=0,
        )
        for step, batch in enumerate(batch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs = classification_model._get_inputs_dict(batch)
            
            loss, *_ = calculate_loss(
                model,
                inputs,
                num_labels=classification_model.num_labels,
                weight=classification_model.weight, 
                device=classification_model.device,
            )

            if args.n_gpu > 1:
                loss = (
                    loss.mean()
                )  # mean() to average on multi-gpu parallel training

            current_loss = loss.item()

            if show_running_loss:
                batch_iterator.set_description(
                    f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.optimizer == "AdamW":
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar(
                        "lr", scheduler.get_last_lr()[0], global_step
                    )
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss
                    if args.wandb_project or classification_model.is_sweeping:
                        wandb.log(
                            {
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            }
                        )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir_current = os.path.join(
                        output_dir, "checkpoint-{}".format(global_step)
                    )

                    classification_model.save_model(
                        output_dir_current, optimizer, scheduler, model=model
                    )

                if args.evaluate_during_training and (
                    args.evaluate_during_training_steps > 0
                    and global_step % args.evaluate_during_training_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results, _, _ = classification_model.eval_model(
                        eval_df,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )

                    output_dir_current = os.path.join(
                        output_dir, "checkpoint-{}".format(global_step)
                    )

                    if args.save_eval_checkpoints:
                        classification_model.save_model(
                            output_dir_current,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )

                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["train_loss"].append(current_loss)
                    for key in results:
                        training_progress_scores[key].append(results[key])

                    if test_df is not None:
                        test_results, _, _ = classification_model.eval_model(
                            test_df,
                            verbose=verbose
                            and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )
                        for key in test_results:
                            training_progress_scores["test_" + key].append(
                                test_results[key]
                            )

                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(
                        os.path.join(
                            args.output_dir, "training_progress_scores.csv"
                        ),
                        index=False,
                    )

                    if args.wandb_project or classification_model.is_sweeping:
                        wandb.log(classification_model._get_last_metrics(training_progress_scores))

                    for key, value in flatten_results(
                        classification_model._get_last_metrics(training_progress_scores)
                    ).items():
                        try:
                            tb_writer.add_scalar(key, value, global_step)
                        except (NotImplementedError, AssertionError):
                            if verbose:
                                logger.warning(
                                    f"can't log value of type: {type(value)} to tensorboar"
                                )
                    tb_writer.flush()

                    if not best_eval_metric:
                        best_eval_metric = results[args.early_stopping_metric]
                        classification_model.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                    if best_eval_metric and args.early_stopping_metric_minimize:
                        if (
                            best_eval_metric - results[args.early_stopping_metric]
                            > args.early_stopping_delta
                        ):
                            best_eval_metric = results[args.early_stopping_metric]
                            classification_model.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if (
                                    early_stopping_counter
                                    < args.early_stopping_patience
                                ):
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(
                                            f" No improvement in {args.early_stopping_metric}"
                                        )
                                        logger.info(
                                            f" Current step: {early_stopping_counter}"
                                        )
                                        logger.info(
                                            f" Early stopping patience: {args.early_stopping_patience}"
                                        )
                                else:
                                    if verbose:
                                        logger.info(
                                            f" Patience of {args.early_stopping_patience} steps reached"
                                        )
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not classification_model.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    else:
                        if (
                            results[args.early_stopping_metric] - best_eval_metric
                            > args.early_stopping_delta
                        ):
                            best_eval_metric = results[args.early_stopping_metric]
                            classification_model.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if (
                                    early_stopping_counter
                                    < args.early_stopping_patience
                                ):
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(
                                            f" No improvement in {args.early_stopping_metric}"
                                        )
                                        logger.info(
                                            f" Current step: {early_stopping_counter}"
                                        )
                                        logger.info(
                                            f" Early stopping patience: {args.early_stopping_patience}"
                                        )
                                else:
                                    if verbose:
                                        logger.info(
                                            f" Patience of {args.early_stopping_patience} steps reached"
                                        )
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not classification_model.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    classification_model.train()

        epoch_number += 1
        output_dir_current = os.path.join(
            output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
        )

        if args.save_model_every_epoch or args.evaluate_during_training:
            os.makedirs(output_dir_current, exist_ok=True)

        if args.save_model_every_epoch:
            classification_model.save_model(output_dir_current, optimizer, scheduler, model=model)

        if args.evaluate_during_training and args.evaluate_each_epoch:
            results, _, _ = classification_model.eval_model(
                eval_df,
                verbose=verbose and args.evaluate_during_training_verbose,
                silent=args.evaluate_during_training_silent,
                wandb_log=False,
                **kwargs,
            )

            classification_model.save_model(
                output_dir_current, optimizer, scheduler, results=results
            )

            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            if test_df is not None:
                test_results, _, _ = classification_model.eval_model(
                    test_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )
                for key in test_results:
                    training_progress_scores["test_" + key].append(
                        test_results[key]
                    )

            report = pd.DataFrame(training_progress_scores)
            report.to_csv(
                os.path.join(args.output_dir, "training_progress_scores.csv"),
                index=False,
            )

            if args.wandb_project or classification_model.is_sweeping:
                wandb.log(classification_model._get_last_metrics(training_progress_scores))

            for key, value in flatten_results(
                classification_model._get_last_metrics(training_progress_scores)
            ).items():
                try:
                    tb_writer.add_scalar(key, value, global_step)
                except (NotImplementedError, AssertionError):
                    if verbose:
                        logger.warning(
                            f"can't log value of type: {type(value)} to tensorboar"
                        )
            tb_writer.flush()

            if not best_eval_metric:
                best_eval_metric = results[args.early_stopping_metric]
                classification_model.save_model(
                    args.best_model_dir,
                    optimizer,
                    scheduler,
                    model=model,
                    results=results,
                )
            if best_eval_metric and args.early_stopping_metric_minimize:
                if (
                    best_eval_metric - results[args.early_stopping_metric]
                    > args.early_stopping_delta
                ):
                    best_eval_metric = results[args.early_stopping_metric]
                    classification_model.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                    early_stopping_counter = 0
                else:
                    if (
                        args.use_early_stopping
                        and args.early_stopping_consider_epochs
                    ):
                        if early_stopping_counter < args.early_stopping_patience:
                            early_stopping_counter += 1
                            if verbose:
                                logger.info(
                                    f" No improvement in {args.early_stopping_metric}"
                                )
                                logger.info(
                                    f" Current step: {early_stopping_counter}"
                                )
                                logger.info(
                                    f" Early stopping patience: {args.early_stopping_patience}"
                                )
                        else:
                            if verbose:
                                logger.info(
                                    f" Patience of {args.early_stopping_patience} steps reached"
                                )
                                logger.info(" Training terminated.")
                                train_iterator.close()
                            return (
                                global_step,
                                tr_loss / global_step
                                if not classification_model.args.evaluate_during_training
                                else training_progress_scores,
                            )
            else:
                if (
                    results[args.early_stopping_metric] - best_eval_metric
                    > args.early_stopping_delta
                ):
                    best_eval_metric = results[args.early_stopping_metric]
                    classification_model.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                    early_stopping_counter = 0
                else:
                    if (
                        args.use_early_stopping
                        and args.early_stopping_consider_epochs
                    ):
                        if early_stopping_counter < args.early_stopping_patience:
                            early_stopping_counter += 1
                            if verbose:
                                logger.info(
                                    f" No improvement in {args.early_stopping_metric}"
                                )
                                logger.info(
                                    f" Current step: {early_stopping_counter}"
                                )
                                logger.info(
                                    f" Early stopping patience: {args.early_stopping_patience}"
                                )
                        else:
                            if verbose:
                                logger.info(
                                    f" Patience of {args.early_stopping_patience} steps reached"
                                )
                                logger.info(" Training terminated.")
                                train_iterator.close()
                            return (
                                global_step,
                                tr_loss / global_step
                                if not classification_model.args.evaluate_during_training
                                else training_progress_scores,
                            )

    return (
        global_step,
        tr_loss / global_step
        if not classification_model.args.evaluate_during_training
        else training_progress_scores,
    )