import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
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

        eval_dataset = classification_model.load_and_cache_examples(
            eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )
        flops = 0

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


                #flops += ModuleUtilsMixin.floating_point_ops(inputs) 
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
        print(f'Number of FLOPs: {flops}')

        return preds, model_outputs