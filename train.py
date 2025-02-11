import argparse
import os
import wandb
import pathlib
from dotenv import load_dotenv
from model_handler import (
    get_model_run_str,
    get_lora_config,
    load_model_lora,
    CustomSFTTrainer
)
from data import get_train_dataset
from peft import get_peft_model
from trl import SFTConfig

load_dotenv()

dataset_path = "glaiveai/glaive-function-calling-v2"
temp_dir = 'dataset_dir'
output_dir = "results"
lora_alpha = 16
max_seq_length = 512
train_eval_split = 0.1
train_test_split = 0.01
seed = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("--use_qlora", action="store_true", default=False, help="Whether to use qlora (default: False)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA R parameter (default: 8)")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct", help="Model name (default: HuggingFaceTB/SmolLM2-360M-Instruct)")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 (default: False)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--max_steps", type=int, default=301, help="Maximum training steps (default: 301)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing (default: False)")
    parser.add_argument("--no-tf32", action="store_false", dest="tf32", default=True, help="Disable tf32 (default: enabled)")
    parser.add_argument("--torch_compile", action="store_true", default=False, help="Enable torch compile (default: False)")
    parser.add_argument("--dataset_size", type=str, default="small", help="Dataset size string (default: small)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    use_qlora = args.use_qlora
    lora_r = args.lora_r
    model_name = args.model_name
    bf16 = args.bf16
    num_train_epochs = args.num_train_epochs
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing
    tf32 = args.tf32
    torch_compile = args.torch_compile
    dataset_size = args.dataset_size

    run_name = get_model_run_str(model_name, lora_r, use_qlora, dataset_size, learning_rate)

    output_dir_final = os.path.join(output_dir, run_name)

    if os.path.exists(output_dir_final):
        print(f"Output directory {output_dir_final} already exists. Skipping training.")
        exit()
        
    wandb.login(key=os.environ["WANDB_API"])
    lora_config = get_lora_config(lora_r, lora_alpha, use_qlora)
    model, tokenizer = load_model_lora(model_name, max_seq_length, use_qlora, use_flash_attention=True)

    dataset_train_eval = get_train_dataset(
        dataset_path, temp_dir, dataset_size,
        train_eval_split, train_test_split, seed
    )

    print('load peft')
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    print("Creating trainer...")
    pathlib.Path(output_dir_final).mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        dataset_text_field="messages",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        tf32=tf32,
        dataloader_pin_memory=False, # pin data to memory
        torch_compile=torch_compile,
        warmup_steps=50,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # 'linear'
        weight_decay=0.01,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        logging_steps=10,
        output_dir=output_dir_final,
        optim="paged_adamw_8bit",
        remove_unused_columns=True,
        seed=seed,
        run_name=run_name,
        report_to="wandb",
        # report_to="none",
        push_to_hub=False,
        eval_steps=25,
    )

    tokenizer.padding_side = 'right'
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_eval["train"],
        eval_dataset=dataset_train_eval["test"],
        processing_class=tokenizer,
        peft_config=lora_config
    )

    print("Training...")
    trainer.train()
