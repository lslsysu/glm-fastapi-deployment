import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/huggingface/'

import subprocess
import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple

import numpy as np

from util import sample_requests, get_tokenizer

file_path = "ShareGPT_V3_unfiltered_cleaned_split.json"
url = "https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

if not os.path.exists(file_path):
    print(f"dataset not found, downloading...")
    subprocess.run(["curl", "-L", "-o", file_path, url], check=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Tuple[prompt_len, completion_len, request_time_in_milliseconds]
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# API_KEY = 'your_api_key'
API_URL = 'http://localhost:8000/v1/chat/completions'
MODEL_UID = 'THUDM/chatglm2-6b' # 填什么都可以，因为是本地部署的模型，但为了规范最好填我们部署的chatglm2-6b

HEADERS = {
    'Content-Type': 'application/json',
    # 'Authorization': f'Bearer {API_KEY}'
}


async def send_request(session, payload, prompt_len):
    request_start_time = time.time()
    async with session.post(API_URL, json=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            # 如果是自部署的model，要注意在响应体中加入completion_tokens词条
            completion_tokens = result["completion_tokens"]
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            return {'error': response.status, 'message': await response.text()}


class BenchMarkRunner:

    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, completion_len
        concurrency: int,
    ):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)
        self.request_queue = asyncio.Queue(100) # 设置大一点或者不设置，否则会阻塞

    async def run(self):
        tasks = []
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))
        # print("finished create_tasks")
        for req in self.requests:
            await self.request_queue.put(req)
        # print("finished request_queue put")
        # When all request is done, most worker will hang on self.request_queue, but at least one worker will exit
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def worker(self):
        # print("Worker started")
        timeout = aiohttp.ClientTimeout(total=5 * 60) # 如果超时，可以尝试调高到 10*60
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self.request_left > 0:
                # print(f"Waiting to get task from queue... (requests left: {self.request_left})")
                prompt, prompt_len, completion_len = await self.request_queue.get()
                # print(f"Got task: prompt_len={prompt_len}, completion_len={completion_len}")
                payload = {
                    'model': MODEL_UID,
                    'messages': [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "top_p": 1.0,
                    'max_length': 8192,
                    'stream': False
                }
                # print("Sending request...")
                response = await send_request(session, payload, prompt_len)
                # print("Request completed")
                self.request_left -= 1
                print(f"Response {len(self.requests) - self.request_left}: {json.dumps(response, ensure_ascii=False, indent=2)}")


def main():
    dataset_path = r'ShareGPT_V3_unfiltered_cleaned_split.json'
    tokenizer_name_or_path = 'THUDM/chatglm2-6b'
    num_request = 100
    concurrency = 5 # 太高有时候会超时（5 min）
    logger.info("Preparing for benchmark.")
    tokenizer = get_tokenizer(tokenizer_name_or_path, trust_remote_code=True, use_fast=True)
    input_requests = sample_requests(dataset_path, num_request, tokenizer)

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")
    throughput = (
            sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()