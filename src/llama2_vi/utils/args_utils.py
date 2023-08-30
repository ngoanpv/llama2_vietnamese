import os
import sys
import torch
import datasets
import transformers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from llama2_vi.config.dataset_arguments import DataArguments
from llama2_vi.config.finetuning_arguments import FinetuningArguments
from llama2_vi.config.generate_arguments import GeneratingArguments
from llama2_vi.config.model_arguments import ModelArguments


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()

def parse_infer_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        FinetuningArguments,
        GeneratingArguments
    ))
    return _parse_args(parser, args)

def get_infer_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    model_args, data_args, finetuning_args, generating_args = parse_infer_args(args)

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            if len(model_args.checkpoint_dir) != 1:
                raise ValueError("Only LoRA tuning accepts multiple checkpoints.")
        elif model_args.quantization_bit is not None and len(model_args.checkpoint_dir) != 1:
                raise ValueError("Quantized model only accepts a single checkpoint.")

    return model_args, data_args, finetuning_args, generating_args
