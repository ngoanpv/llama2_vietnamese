# Llama2 Vietnamese

<p align="center">
  <img src="docs/imgs/logo.png" alt="Llama 2 Logo" width="200"/>
</p>

Mô hình ngôn ngữ lớn được tinh chỉnh cho tiếng Việt dựa trên mô hình Llama 2.

## Giới thiệu

Mục đích của project này là để thực nghiệm các mô hình LLM cho tiếng Việt, bắt đầu với các bước fine-tuning và sẽ mở rộng đến pre-training nếu resource cho phép.

## Tiến độ hiện tại
### 30/08/2023

Mô hình finetuned trên Llama 2 7B (https://huggingface.co/meta-llama/Llama-2-7b-hf) với dữ liệu chứa 20k câu hỏi đáp. Đây là bước thực nghiệm ban đầu và sẽ mở rộng tiếp sau đó. 

## Model Checkpoint

Checkpoint của model có thể tìm thấy tại HuggingFace [ở đây](https://huggingface.co/ngoantech/Llama-2-7b-vietnamese-20k).

## Ví dụ về kết quả của model

<img src="docs/imgs/exp_1.png" alt="output_1"/>  


## Các bước để bắt đầu

1. Clone the repository:
    ```bash
    git clone https://github.com/ngoanpv/llama2_vietnamese
    cd llama2_vietnamese
    ```

2. Cài đặt các packages cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

3. Khởi động FastAPI server, hiện tại có 2 APIs:
    ```bash
    python serving/fastapi/main.py
    ```

4. Sử dụng script dưới đây để test 2 API trên:
    ```bash
    python scripts/request_fastapi.py
    ```


## Kế hoạch tiếp theo

- Fine tune trên bộ dữ liệu lớn hơn
- Thử nghiệm điều chỉnh tokenizer và các bước cho pre-training



