| Parameter | Description |
|-----------|------------|
| **use_qlora** | Whether to use **QLoRA (4-bit quantization)** for training. |
| **lora_r** | LoRA rank controls how many parameters will be injected (`4`, `8`, `16`) |
| **model_name** | Either `SmolLM2-135M-Instruct` or `SSmolLM2-360M-Instruct`. |
| **learning_rate** | Learning rate values (`1e-4`, `2e-4`, `4e-4`). |
| **dataset_size** | Control the amount of samples (`small` or `medium`) |
| **max_steps** | Number of training steps - fixed as `350` steps. |
| **gradient_checkpointing** | Whether to enable gradient checkpointing - fixed as `True` to optimize resource usage |
| **tf32 & bf16** | Controls mixed precision - fixed as `True` to optimize resource usage |