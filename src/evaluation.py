import logging
import os
import warnings
from collections import Counter
from dataclasses import asdict
from sklearn import metrics
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             label_ranking_average_precision_score,
                             matthews_corrcoef, roc_curve)
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.classification.classification_utils import InputExample
from src.utils import calculate_loss

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

def eval_model(
    classification_model,
    eval_df,
    multi_label=True,
    output_dir=None,
    verbose=True,
    silent=False,
    wandb_log=True,
    **kwargs,
):
    """
    Evaluates the model on eval_df. Saves results to output_dir.

    Args:
        eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
        the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
        output_dir: The directory where model files will be saved. If not given, classification_model.args.output_dir will be used.
        verbose: If verbose, results will be printed to the console on completion of evaluation.
        silent: If silent, tqdm progress bars will be hidden.
        wandb_log: If True, evaluation results will be logged to wandb.
        **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                    A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

    Returns:
        result: Dictionary containing evaluation results.
        model_outputs: List of model outputs for each row in eval_df
        wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
    """  # noqa: ignore flake8"

    if not output_dir:
        output_dir = classification_model.args.output_dir

    classification_model._move_model_to_device()

    result, model_outputs, wrong_preds = evaluate(
        classification_model,
        eval_df,
        output_dir,
        multi_label=multi_label,
        verbose=verbose,
        silent=silent,
        wandb_log=wandb_log,
        **kwargs,
    )
    classification_model.results.update(result)

    if verbose:
        logger.info(classification_model.results)

    return result, model_outputs, wrong_preds

def evaluate(
    classification_model,
    eval_df,
    output_dir,
    multi_label=True,
    prefix="",
    verbose=True,
    silent=False,
    wandb_log=True,
    **kwargs,
):
    """
    Evaluates the model on eval_df.

    Utility function to be used by the eval_model() method. Not intended to be used directly.
    """

    model = classification_model.model
    args = classification_model.args
    eval_output_dir = output_dir

    results = {}

    if "text" in eval_df.columns and "labels" in eval_df.columns:
        eval_examples = (
            eval_df["text"].astype(str).tolist(),
            eval_df["labels"].tolist(),
        )
    else:
        warnings.warn(
            "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
        )
        eval_examples = (
            eval_df.iloc[:, 0].astype(str).tolist(),
            eval_df.iloc[:, 1].tolist(),
        )


    eval_dataset = classification_model.load_and_cache_examples(
        eval_examples, evaluate=True, verbose=verbose, silent=silent
    )
    os.makedirs(eval_output_dir, exist_ok=True)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(eval_dataloader)
    preds = np.empty((len(eval_dataset), classification_model.num_labels))
    if multi_label:
        out_label_ids = np.empty((len(eval_dataset), classification_model.num_labels))
    else:
        out_label_ids = np.empty((len(eval_dataset)))
    model.eval()

    for i, batch in enumerate(
        tqdm(
            eval_dataloader,
            disable=args.silent or silent,
            desc="Running Evaluation",
        )
    ):
        # batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = classification_model._get_inputs_dict(batch)
            outputs = calculate_loss(
                model,
                inputs,
                num_labels=classification_model.num_labels,
                weight=classification_model.weight, 
                device=classification_model.device,
            )
            tmp_eval_loss, logits = outputs[:2]

            if multi_label:
                logits = logits.sigmoid()
            if classification_model.args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        start_index = classification_model.args.eval_batch_size * i
        end_index = (
            start_index + classification_model.args.eval_batch_size
            if i != (n_batches - 1)
            else len(eval_dataset)
        )
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = (
            inputs["labels"].detach().cpu().numpy()
        )

    eval_loss = eval_loss / nb_eval_steps

    if not multi_label and args.regression is True:
        preds = np.squeeze(preds)
        model_outputs = preds
    else:
        model_outputs = preds 
        if not multi_label:
            preds = np.argmax(preds, axis=1)

    result, wrong = compute_metrics(
        classification_model, preds, model_outputs, out_label_ids, eval_examples, **kwargs
    )
    result["eval_loss"] = eval_loss
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))

    if (
        classification_model.args.wandb_project
        and wandb_log
        and not multi_label
        and not classification_model.args.regression
    ):
        if not wandb.setup().settings.sweep_id:
            logger.info(" Initializing WandB run for evaluation.")
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="comparison_transformers")
        if not args.labels_map:
            classification_model.args.labels_map = {i: i for i in range(classification_model.num_labels)}

        labels_list = sorted(list(classification_model.args.labels_map.keys()))
        inverse_labels_map = {
            value: key for key, value in classification_model.args.labels_map.items()
        }

        truth = [inverse_labels_map[out] for out in out_label_ids]

        # Confusion Matrix
        wandb.sklearn.plot_confusion_matrix(
            truth,
            [inverse_labels_map[pred] for pred in preds],
            labels=labels_list,
        )

        if not classification_model.args.sliding_window:
        # ROC`
            wandb.log({"roc": wandb.plots.ROC(truth, model_outputs, labels_list)})

            # Precision Recall
            wandb.log(
                {
                    "pr": wandb.plots.precision_recall(
                        truth, model_outputs, labels_list
                    )
                }
            )
        
        
    if (
        classification_model.args.wandb_project
        and wandb_log
        and multi_label
    ):
        if not wandb.setup().settings.sweep_id:
            logger.info(" Initializing WandB run for evaluation.")
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="comparison_transformers")
        # log to wandb
        wandb.log(result)

    return results, model_outputs, wrong

def compute_metrics(
        classification_model,
        preds,
        model_outputs,
        labels,
        eval_examples=None,
        multi_label=True,
        **kwargs,
    ):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            model_outputs: Model outputs
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            For non-binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn).
            For binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn, AUROC, AUPRC).
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            if metric.startswith("prob_"):
                extra_metrics[metric] = func(labels, model_outputs)
            else:
                extra_metrics[metric] = func(labels, preds)

        if multi_label:
            threshold_values = classification_model.args.threshold if classification_model.args.threshold else 0.5
            if isinstance(threshold_values, list):
                preds_thresholded = [
                    [
                        classification_model._threshold(pred, threshold_values[i])
                        for i, pred in enumerate(example)
                    ]
                    for example in preds
                ]
                mismatched = labels != preds_thresholded
            else:
                preds_thresholded = [
                    [classification_model._threshold(pred, threshold_values) for pred in example]
                    for example in preds
                ] 
                mismatched = labels != preds_thresholded
        else:
            mismatched = labels != preds
        


        if eval_examples:
            if not isinstance(eval_examples[0], InputExample):
                
                # Single sentence task
                eval_examples = [
                    InputExample(
                        guid=i,
                        text_a=example,
                        text_b=None,
                        label=label,
                    )
                    for i, (example, label) in enumerate(
                        zip(eval_examples[0], eval_examples[1])
                    )
                ]
                
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            f1_score_avg = metrics.f1_score(labels, preds_thresholded, average='samples') 
            f1_score_macro = metrics.f1_score(labels, preds_thresholded, average='macro')
            f1_score_micro = metrics.f1_score(labels, preds_thresholded, average='micro')

            return {**{"LRAP": label_ranking_score, "f1_score_avg": f1_score_avg, "f1_score_macro": f1_score_macro, "f1_score_micro": f1_score_micro }, **extra_metrics}, wrong
        elif classification_model.args.regression:
            return {**extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)
        if classification_model.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            if classification_model.args.sliding_window:
                return (
                    {
                        **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                        **extra_metrics,
                    },
                    wrong,
                )
            else:
                scores = np.array([softmax(element)[1] for element in model_outputs])
                fpr, tpr, thresholds = roc_curve(labels, scores)
                auroc = auc(fpr, tpr)
                auprc = average_precision_score(labels, scores)
                return (
                    {
                        **{
                            "mcc": mcc,
                            "tp": tp,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "auroc": auroc,
                            "auprc": auprc,
                        },
                        **extra_metrics,
                    },
                    wrong,
                )
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong