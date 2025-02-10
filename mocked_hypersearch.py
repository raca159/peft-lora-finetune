import os
import tqdm

use_qlora = [False, True]
lora_rv= [4, 8, 16]
model_name = ["HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"]
bf16 = [True]
max_steps = [500]
learning_rates = [1e-4, 2e-4, 4e-4]
gradient_checkpointing = [True]
tf32 = [True]
dataset_size = ["small", "medium"]

if __name__ == '__main__':
    cmds = []
    # Create a hyperparameter search
    for qloranow in use_qlora:
        for lora_r in lora_rv:
            for model_name in model_name:
                for bf16now in bf16:
                    for max_stepsnow in max_steps:
                        for learning_rate in learning_rates:
                            for gradient_checkpointingnow in gradient_checkpointing:
                                for tf32now in tf32:
                                    for dataset_size in dataset_size:
                                        # Run the training script with the set of hyperparameters
                                        cmd = f'python train.py --lora_r {lora_r} --model_name {model_name} --max_steps {max_stepsnow} --learning_rate {learning_rate} --dataset_size {dataset_size}'
                                        if qloranow:
                                            cmd += " --use_qlora"
                                        if gradient_checkpointingnow:
                                            cmd += " --gradient_checkpointing"
                                        if not tf32now:
                                            cmd += " --no-tf32"
                                        if not bf16now:
                                            cmd += " --bf16"
                                        cmds.append(cmd)
    for cmd in tqdm.tqdm(cmds, desc="Running hyperparameter search"):
        os.system(cmd)