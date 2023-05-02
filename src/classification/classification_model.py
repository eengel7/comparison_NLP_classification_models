#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import collections
import logging
import os
import random
import tempfile
import warnings
from pathlib import Path

from scipy.stats import mode
from tqdm.auto import tqdm
from src.utils import calculate_loss
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import WEIGHTS_NAME  # NystromformerTokenizer,
from transformers import (AlbertConfig, AlbertForSequenceClassification,
                          AlbertTokenizer, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, BertForSequenceClassification,
                          BertTokenizerFast, BertweetTokenizer, BigBirdConfig,
                          BigBirdForSequenceClassification, BigBirdTokenizer,
                          CamembertConfig, CamembertForSequenceClassification,
                          CamembertTokenizerFast, DebertaConfig,
                          DebertaForSequenceClassification, DebertaTokenizer,
                          DebertaV2Config, DebertaV2ForSequenceClassification,
                          DebertaV2Tokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizerFast, ElectraConfig,
                          ElectraForSequenceClassification,
                          ElectraTokenizerFast, FlaubertConfig,
                          FlaubertForSequenceClassification, FlaubertTokenizer,
                          HerbertTokenizerFast, LongformerConfig,
                          LongformerForSequenceClassification,
                          LongformerTokenizerFast, MobileBertConfig,
                          MobileBertForSequenceClassification,
                          MobileBertTokenizerFast, MPNetConfig,
                          MPNetForSequenceClassification, MPNetTokenizerFast,
                          NystromformerConfig,
                          NystromformerForSequenceClassification,
                          RemBertConfig, RemBertForSequenceClassification,
                          RemBertTokenizerFast, RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizerFast, SqueezeBertConfig,
                          SqueezeBertForSequenceClassification,
                          SqueezeBertTokenizerFast, XLMConfig,
                          XLMForSequenceClassification, XLMRobertaConfig,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizerFast, XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification, XLNetTokenizerFast)
from transformers.convert_graph_to_onnx import convert, quantize

from config.model_args import ClassificationArgs
from config.utils import sweep_config_to_sweep_values
from src.classification.classification_utils import (
    ClassificationDataset, InputExample, convert_examples_to_features)

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT = ["squeezebert", "deberta", "mpnet"]

MODELS_WITH_EXTRA_SEP_TOKEN = [
    "roberta",
    "camembert",
    "xlmroberta",
    "longformer",
    "mpnet",
    "nystromformer",
]

MODELS_WITH_ADD_PREFIX_SPACE = [
    "roberta",
    "camembert",
    "xlmroberta",
    "longformer",
    "mpnet",
    "nystromformer",
]


class ClassificationModel:
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda= False,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
    ):

        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a string is passed, Simple Transformers will try to initialize a tokenizer class from the available MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
            tokenizer_name: The name/path to the tokenizer. If the tokenizer_type is not specified, the model_type will be used to determine the type of the tokenizer.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
            "bertweet": (
                RobertaConfig,
                RobertaForSequenceClassification,
                BertweetTokenizer,
            ),
            "bigbird": (
                BigBirdConfig,
                BigBirdForSequenceClassification,
                BigBirdTokenizer,
            ),
            "camembert": (
                CamembertConfig,
                CamembertForSequenceClassification,
                CamembertTokenizerFast,
            ),
            "deberta": (
                DebertaConfig,
                DebertaForSequenceClassification,
                DebertaTokenizer,
            ),
            "debertav2": (
                DebertaV2Config,
                DebertaV2ForSequenceClassification,
                DebertaV2Tokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            ),
            "electra": (
                ElectraConfig,
                ElectraForSequenceClassification,
                ElectraTokenizerFast,
            ),
            "flaubert": (
                FlaubertConfig,
                FlaubertForSequenceClassification,
                FlaubertTokenizer,
            ),
            "herbert": (
                BertConfig,
                BertForSequenceClassification,
                HerbertTokenizerFast,
            ),
            "longformer": (
                LongformerConfig,
                LongformerForSequenceClassification,
                LongformerTokenizerFast,
            ),
            "mobilebert": (
                MobileBertConfig,
                MobileBertForSequenceClassification,
                MobileBertTokenizerFast,
            ),
            "mpnet": (MPNetConfig, MPNetForSequenceClassification, MPNetTokenizerFast),
            "nystromformer": (
                NystromformerConfig,
                NystromformerForSequenceClassification,
                BigBirdTokenizer,
            ),
            "rembert": (
                RemBertConfig,
                RemBertForSequenceClassification,
                RemBertTokenizerFast,
            ),
            "roberta": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
            ),
            "squeezebert": (
                SqueezeBertConfig,
                SqueezeBertForSequenceClassification,
                SqueezeBertTokenizerFast,
            ),
            "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForSequenceClassification,
                XLMRobertaTokenizerFast,
            ),
            "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        #Sets the number of threads used for intraop parallelism on CPU.
        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.labels_list and not self.args.lazy_loading:
            if num_labels:
                assert num_labels == len(self.args.labels_list)
            if self.args.labels_map:
                try:
                    assert list(self.args.labels_map.keys()) == self.args.labels_list
                except AssertionError:
                    assert [
                        int(key) for key in list(self.args.labels_map.keys())
                    ] == self.args.labels_list
                    self.args.labels_map = {
                        int(key): value for key, value in self.args.labels_map.items()
                    }
            else:
                self.args.labels_map = {
                    label: i for i, label in enumerate(self.args.labels_list)
                }
        else:
            len_labels_list = 2 if not num_labels else num_labels
            self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.model = model_class.from_pretrained(
            model_name, config=self.config, **kwargs
        )
        

        self.results = {}

        if tokenizer_name is None:
            tokenizer_name = model_name

        if tokenizer_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

    def load_and_cache_examples(
        self,
        examples,
        evaluate=False,
        no_cache=False,
        multi_label=False,
        verbose=True,
        silent=False,
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        dataset = ClassificationDataset(
            examples,
            self.tokenizer,
            self.args,
            mode=mode,
            multi_label=multi_label,
            output_mode=output_mode,
            no_cache=no_cache,
        )
        return dataset
    
    def get_floating_point_ops(self, ):

        return None


    def predict(self, to_predict, multi_label=False):
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

            model = self.model
            args = self.args

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = np.empty((len(to_predict), self.num_labels))
            if multi_label:
                out_label_ids = np.empty((len(to_predict), self.num_labels))
            else:
                out_label_ids = np.empty((len(to_predict)))

            if not multi_label and self.args.onnx:
                model_inputs = self.tokenizer.batch_encode_plus(
                    to_predict, return_tensors="pt", padding=True, truncation=True
                )

                if self.args.model_type in [
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
                        output = self.model.run(None, inputs_onnx)

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
                        output = self.model.run(None, inputs_onnx)

                        preds[i] = output[0]

                model_outputs = preds
                preds = np.argmax(preds, axis=1)

            else:
                self._move_model_to_device()
                dummy_label = (
                    0
                    if not self.args.labels_map
                    else next(iter(self.args.labels_map.keys()))
                )

                if multi_label:
                    dummy_label = [dummy_label for i in range(self.num_labels)]

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

                
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
                )

                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(
                    eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
                )

                n_batches = len(eval_dataloader)
                for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent)):
                    model.eval()
                    # batch = tuple(t.to(device) for t in batch)

                    with torch.no_grad():
                        inputs = self._get_inputs_dict(batch, no_hf=True)
                        outputs = calculate_loss(
                            model,
                            inputs,
                            weight=self.weight,
                            num_labels=self.num_labels,
                            device=self.device,
                        )
                        tmp_eval_loss, logits = outputs[:2]

                        if multi_label:
                            logits = logits.sigmoid()

                        if self.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    start_index = self.args.eval_batch_size * i
                    end_index = (
                        start_index + self.args.eval_batch_size
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
                                    self._threshold(pred, threshold_values[i])
                                    for i, pred in enumerate(example)
                                ]
                                for example in preds
                            ]
                        else:
                            preds = [
                                [self._threshold(pred, args.threshold) for pred in example]
                                for example in preds
                            ]
                    else:
                        preds = np.argmax(preds, axis=1)

            if self.args.labels_map and not self.args.regression:
                inverse_labels_map = {
                    value: key for key, value in self.args.labels_map.items()
                }
                preds = [inverse_labels_map[pred] for pred in preds]


            return preds, model_outputs


    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(
                    output_dir
                )
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="sentiment-analysis",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self.save_model_args(output_dir)



    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch, no_hf=False):
        if self.args.use_hf_datasets and not no_hf:
            return {key: value.to(self.device) for key, value in batch.items()}
        if isinstance(batch[0], dict) or isinstance(batch[0].data, dict):
            inputs = {
                key: value.squeeze(1).to(self.device) for key, value in batch[0].items()
            }
            inputs["labels"] = batch[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2]
                    if self.args.model_type
                    in ["bert", "xlnet", "albert", "layoutlm", "layoutlmv2"]
                    else None
                )

        return inputs
    
    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = ClassificationArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]