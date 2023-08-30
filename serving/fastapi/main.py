from fastapi import FastAPI, Depends, Query
from fastapi.responses import StreamingResponse
import os
import torch
from typing import Optional, List, Tuple, Generator
from threading import Thread

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llama2_vi.inference.chat import ChatModel

app = FastAPI()

chatmodel = ChatModel({
    'model_name_or_path':'ngoantech/Llama-2-7b-vietnamese-20k',
    'template':'llama2_vi',
    'finetuning_type':'lora',
    'temperature': 0.1
})

@app.get("/chat/")
def chat_with_model(question: str):
    return {"response": chatmodel.chat(question)}

@app.get("/stream_chat/", response_class=StreamingResponse)
async def stream_chat_endpoint(
    question: str
):
    generator = chatmodel.stream_chat(question)
    return StreamingResponse(generator, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8686)
