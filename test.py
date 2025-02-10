import os
import pathlib
from peft import get_peft_model
from trl import SFTConfig, SFTTrainer
import wandb
from typing import Optional, Callable
import warnings
import multiprocessing
import os
import torch
import re
from functools import partial
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from peft import PeftModel


chat_interaction = [
    {
        "role": "system",
        "content": '''You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "convert_currency",
    "description": "Convert the amount from one currency to another",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "The amount to convert"
            },
            "from_currency": {
                "type": "string",
                "description": "The currency to convert from"
            },
            "to_currency": {
                "type": "string",
                "description": "The currency to convert to"
            }
        },
        "required": [
            "amount",
            "from_currency",
            "to_currency"
        ]
    }
}'''
},
    {
        "role": "user",
        "content": "Hi, I need to convert 500 US dollars to Euros. Can you help me with that?"
    }
]
prompt = tokenizer.apply_chat_template(chat_interaction, tokenize=False, add_generation_prompt=True)
inf_peft_model = PeftModel.from_pretrained(model, os.path.join(output_dir_final, os.listdir(output_dir_final)[-1]))
print(run_inout_pipe(chat_interaction, tokenizer, inf_peft_model))

