import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/huggingface/'

import time
import torch
import uvicorn
import json
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import schemas

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
app = FastAPI(lifespan=lifespan)

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=schemas.ModelList)
async def list_models():
    global model_args
    model_card = schemas.ModelCard(id="gpt-3.5-turbo")
    return schemas.ModelList(data=[model_card])

async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = schemas.ChatCompletionResponseStreamChoice(
        index=0,
        delta=schemas.DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = schemas.ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}"

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = schemas.ChatCompletionResponseStreamChoice(
            index=0,
            delta=schemas.DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = schemas.ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}"


    choice_data = schemas.ChatCompletionResponseStreamChoice(
        index=0,
        delta=schemas.DeltaMessage(),
        finish_reason="stop"
    )
    chunk = schemas.ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}"
    yield '[DONE]'

# 输入时，schemas.model填什么都无所谓，因为我们不会用到。为了规范最好填chatglm2-6b
@app.post("/v1/chat/completions", response_model=schemas.ChatCompletionResponse)
async def create_chat_completion(request: schemas.ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = schemas.ChatCompletionResponseChoice(
        index=0,
        message=schemas.ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    # 计算回复所用tokens数，用于压力测试
    content_tokens = len(tokenizer.encode(response))

    return schemas.ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", completion_tokens = content_tokens)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)