
FROM huggingface/transformers-pytorch-gpu:latest AS builder

EXPOSE 8888

RUN pip install --no-cache-dir transformers datasets==3.2.0 accelerate==1.3.0 evaluate==0.4.3 bitsandbytes==0.45.1 \
                            peft==0.14.0 trl==0.13.0 wandb==0.19.4 python-dotenv==1.0.1 jupyter==1.1.1 \
                            optimum==1.23.3 flash-attn==2.7.3 langchain==0.3.18 langchain-openai==0.3.5 \
                            pandas==2.2.3

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]