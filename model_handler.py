from trl import SFTTrainer
from typing import Optional, Callable
import warnings
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig


class CustomSFTTrainer(SFTTrainer):

    def _prepare_non_packed_dataloader(
        self,
        processing_class,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        return tokenized_dataset

def get_model_run_str(model_name, lora_r, use_qlora, dataset_size, learning_rate):
    lr = format(learning_rate, '.0e')
    run_name = f"{model_name.split('/')[-1]}-fncall_peft-ds_{dataset_size}-lora_r_{lora_r}-use_qlora_{use_qlora}-lr_{lr}"
    return run_name

def load_model_lora(model_name, max_seq_length, use_qlora=False):
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding=True, truncation=True, max_length=max_seq_length
    )
    return model, tokenizer

def get_lora_config(lora_r=8, lora_alpha=32, use_qlora=False, inference_mode=False):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        inference_mode=inference_mode,
        use_dora=False if use_qlora else True,
    )
    return lora_config

def run_inout_pipe(chat_interaction, tokenizer, model):
    prompt = tokenizer.apply_chat_template(chat_interaction, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    outputs = outputs[:, inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
