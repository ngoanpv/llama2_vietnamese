# Llama2 Vietnamese

<img src="docs/imgs/logo.png" alt="Llama 2 Logo" width="200"/>  

[Read in Vietnamese](README_vi.md)

A fine-tuned Large Language Model (LLM) for the Vietnamese language based on the Llama 2 model.

## Introduction

This project is an effort to bring the power of large language models to the Vietnamese language. 


## Current Status
### 30 Aug 2023

I've just rolled out a experience version of a large language model for Vietnamese (finetuned on Llama2-7b (https://huggingface.co/meta-llama/Llama-2-7b-hf)). This model has been fine-tuned on a 20k instruction data sample. It's experimental and intended for lightweight tasks.

## Model Checkpoint

The model has been published on Huggingface and can be accessed [here](https://huggingface.co/ngoantech/Llama-2-7b-vietnamese-20k).




## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/ngoanpv/llama2_vietnamese
    cd llama2_vietnamese
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the FastAPI server:
    ```bash
    python serving/fastapi/main.py
    ```

4. To test the server, use the provided script:
    ```bash
    python scripts/request_fastapi.py
    ```


## Future Plans

Stay tuned for future releases as we are continuously working on improving the model, expanding the dataset, and adding new features.



