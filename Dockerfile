
FROM huggingface/transformers-pytorch-gpu:latest AS builder

EXPOSE 8888

RUN pip install --no-cache-dir transformers datasets accelerate evaluate bitsandbytes peft trl tensorboard wandb python-dotenv jupyter optimum flash-attn

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]