import numpy as np
import torch
from scipy.stats import mode
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm

from src.utils import calculate_loss


def predict(classification_model, to_predict, multi_label=False):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
                        For layoutlm and layoutlmv2 model types, this should be a list of lists:
                        [
                            [text1, [x0], [y0], [x1], [y1]],
                            [text2, [x0], [y0], [x1], [y1]],
                            ...
                            [textn, [x0], [y0], [x1], [y1]]
                        ]

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        model = classification_model.model
        args = classification_model.args

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = np.empty((len(to_predict), classification_model.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(to_predict), classification_model.num_labels))
        else:
            out_label_ids = np.empty((len(to_predict)))

        if not multi_label and classification_model.args.onnx:
            model_inputs = classification_model.tokenizer.batch_encode_plus(
                to_predict, return_tensors="pt", padding=True, truncation=True
            )

            if classification_model.args.model_type in [
                "bert",
                "xlnet",
                "albert",
                "layoutlm",
                "layoutlmv2",
            ]:
                for i, (input_ids, attention_mask, token_type_ids) in enumerate(
                    zip(
                        model_inputs["input_ids"],
                        model_inputs["attention_mask"],
                        model_inputs["token_type_ids"],
                    )
                ):
                    input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                    attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                    token_type_ids = token_type_ids.unsqueeze(0).detach().cpu().numpy()
                    inputs_onnx = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    }

                    # Run the model (None = get all the outputs)
                    output = classification_model.model.run(None, inputs_onnx)

                    preds[i] = output[0]

            else:
                for i, (input_ids, attention_mask) in enumerate(
                    zip(model_inputs["input_ids"], model_inputs["attention_mask"])
                ):
                    input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                    attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                    inputs_onnx = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }

                    # Run the model (None = get all the outputs)
                    output = classification_model.model.run(None, inputs_onnx)

                    preds[i] = output[0]

            model_outputs = preds
            preds = np.argmax(preds, axis=1)

        else:
            classification_model._move_model_to_device()
            dummy_label = (
                0
                if not classification_model.args.labels_map
                else next(iter(classification_model.args.labels_map.keys()))
            )

            if multi_label:
                dummy_label = [dummy_label for i in range(classification_model.num_labels)]

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if isinstance(to_predict[0], list):
                eval_examples = (
                    *zip(*to_predict),
                    [dummy_label for i in range(len(to_predict))],
                )
            else:
                eval_examples = (
                    to_predict,
                    [dummy_label for i in range(len(to_predict))],
                )

            if args.sliding_window:
                eval_dataset, window_counts = classification_model.load_and_cache_examples(
                    eval_examples, evaluate=True, no_cache=True
                )
                preds = np.empty((len(eval_dataset), classification_model.num_labels))
                if multi_label:
                    out_label_ids = np.empty((len(eval_dataset), classification_model.num_labels))
                else:
                    out_label_ids = np.empty((len(eval_dataset)))
            else:
                eval_dataset = classification_model.load_and_cache_examples(
                    eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
                )

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            if classification_model.config.output_hidden_states:
                model.eval()
                preds = None
                out_label_ids = None
                for i, batch in enumerate(
                    tqdm(
                        eval_dataloader, disable=args.silent, desc="Running Prediction"
                    )
                ):
                    # batch = tuple(t.to(classification_model.device) for t in batch)
                    with torch.no_grad():
                        inputs = classification_model._get_inputs_dict(batch, no_hf=True)

                        outputs = calculate_loss(
                            model,
                            inputs,
                            num_labels=classification_model.num_labels,
                            weight=classification_model.weight, 
                            device=classification_model.device,
                        )
                        tmp_eval_loss, logits = outputs[:2]
                        embedding_outputs, layer_hidden_states = (
                            outputs[2][0],
                            outputs[2][1:],
                        )

                        if multi_label:
                            logits = logits.sigmoid()

                        if classification_model.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs["labels"].detach().cpu().numpy()
                        all_layer_hidden_states = np.array(
                            [
                                state.detach().cpu().numpy()
                                for state in layer_hidden_states
                            ]
                        )
                        all_embedding_outputs = embedding_outputs.detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(
                            out_label_ids,
                            inputs["labels"].detach().cpu().numpy(),
                            axis=0,
                        )
                        all_layer_hidden_states = np.append(
                            all_layer_hidden_states,
                            np.array(
                                [
                                    state.detach().cpu().numpy()
                                    for state in layer_hidden_states
                                ]
                            ),
                            axis=1,
                        )
                        all_embedding_outputs = np.append(
                            all_embedding_outputs,
                            embedding_outputs.detach().cpu().numpy(),
                            axis=0,
                        )
            else:
                n_batches = len(eval_dataloader)
                for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent)):
                    model.eval()
                    # batch = tuple(t.to(device) for t in batch)

                    with torch.no_grad():
                        inputs = classification_model._get_inputs_dict(batch, no_hf=True)
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

            if args.sliding_window:
                count = 0
                window_ranges = []
                for n_windows in window_counts:
                    window_ranges.append([count, count + n_windows])
                    count += n_windows

                preds = [
                    preds[window_range[0] : window_range[1]]
                    for window_range in window_ranges
                ]

                model_outputs = preds
                if args.regression is True:
                    preds = [np.squeeze(pred) for pred in preds]
                    final_preds = []
                    for pred_row in preds:
                        mean_pred = np.mean(pred_row)
                        print(mean_pred)
                        final_preds.append(mean_pred)
                    preds = np.array(final_preds)
                else:
                    preds = [np.argmax(pred, axis=1) for pred in preds]
                    final_preds = []
                    for pred_row in preds:
                        mode_pred, counts = mode(pred_row)
                        if len(counts) > 1 and counts[0] == counts[1]:
                            final_preds.append(args.tie_value)
                        else:
                            final_preds.append(mode_pred[0])
                    preds = np.array(final_preds)
            elif not multi_label and args.regression is True:
                preds = np.squeeze(preds)
                model_outputs = preds
            else:
                model_outputs = preds
                if multi_label:
                    if isinstance(args.threshold, list):
                        threshold_values = args.threshold
                        preds = [
                            [
                                classification_model._threshold(pred, threshold_values[i])
                                for i, pred in enumerate(example)
                            ]
                            for example in preds
                        ]
                    else:
                        preds = [
                            [classification_model._threshold(pred, args.threshold) for pred in example]
                            for example in preds
                        ]
                else:
                    preds = np.argmax(preds, axis=1)

        if classification_model.args.labels_map and not classification_model.args.regression:
            inverse_labels_map = {
                value: key for key, value in classification_model.args.labels_map.items()
            }
            preds = [inverse_labels_map[pred] for pred in preds]

        if classification_model.config.output_hidden_states:
            return preds, model_outputs, all_embedding_outputs, all_layer_hidden_states
        else:
            return preds, model_outputs