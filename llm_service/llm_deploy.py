from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch
import uvicorn
import nvidia_smi
import logging
import traceback
import re
import gc
from string import Template
import datetime
from vllm import LLM, SamplingParams

nvidia_smi.nvmlInit()
GPU_Device_Count = nvidia_smi.nvmlDeviceGetCount()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(module)s@%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
log = logging.getLogger()

app = FastAPI()

# original qwen 0.5b model
model_path = "/home/tz/Desktop/bigai/model/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
    max_tokens=512,
    stop_token_ids=[tokenizer.eos_token_id],
)
llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.7)  # optional


class StatusResponse(BaseModel):
    status_code: int
    input_tokens: int
    output_tokens: int
    delta_t: float
    content: str


def log_gpu_usage(str_template):
    log.info(str_template.substitute(msg=f"Memory Usage..."))
    for i in range(GPU_Device_Count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        log.info(
            str_template.substitute(
                msg=f"|Device {i}| Mem Free: {mem.free/1024**3:5.2f}GB / {mem.total/1024**3:5.2f}GB | gpu-util: {util.gpu/100.0:3.1%} | gpu-mem: {util.memory/100.0:3.1%} |"
            )
        )


@app.post("/generate", response_model=StatusResponse)
async def generate(request: Request) -> StatusResponse:
    request = await request.json()

    session_id = (
        request.get("session_id", "ts") if request.get("session_id", "ts") else "None"
    )
    agent_id = (
        request.get("agent_id", "tt") if request.get("agent_id", "tt") else "None"
    )
    tp = Template(session_id + "," + agent_id + ":$msg")

    try:
        messages = request.get("prompts", [])
        # log.info("messages: %s KB", sys.getsizeof(messages) / 1024)
        # log.info("U messages: %s ", messages[3]["content"])
        # log.info("V messages: %s ", messages[1]["content"])
        if len(messages) == 6:
            log.info(tp.substitute(msg=f"feedback messages: {messages[4]['content']} "))

        # log.info(tp.substitute(msg=f"msg: {messages}"))
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        log_gpu_usage(tp)
        start_time = datetime.datetime.now()
        outputs = llm.generate(input_text, sampling_params)
        end_time = datetime.datetime.now()
        dt = (end_time - start_time).total_seconds()
        log.info(tp.substitute(msg=f"generate time: {dt}s "))
        log_gpu_usage(tp)
        input_size = len(outputs[0].prompt_token_ids)
        log.info(tp.substitute(msg=f"input tokens: {input_size} "))

        output_size = len(outputs[0].outputs[0].token_ids)
        log.info(tp.substitute(msg=f"----------------"))
        log.info(
            tp.substitute(msg=f"output tokens n/s:{output_size} {output_size/dt} ")
        )

        response = outputs[0].outputs[0].text
        log.info(tp.substitute(msg=f"response: {response}"))

        # remove char around {}
        pre_idx = 0
        for i, v in enumerate(response):
            if v == "{":
                pre_idx = i
                break
        response = response[pre_idx:]
        suf_idx = 0
        for i, v in enumerate(reversed(response)):
            if v == "}":
                suf_idx = i
                break
        response = response[: len(response) - suf_idx]

        # removes trailing commas in JSON strings
        regex = r"""(?<=[}\]"']),(?!\s*[{["'])"""
        response = re.sub(regex, "", response, 0)

        del request, messages
        del outputs

        current_minute = datetime.datetime.now().minute
        if current_minute % 2 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        return StatusResponse(
            status_code=0,
            input_tokens=input_size,
            output_tokens=output_size,
            content=response,
            delta_t=dt,
        )
    except Exception as e:
        log.info(tp.substitute(msg=f"error: {str(e)}"))
        traceback.print_exc()
        return StatusResponse(
            status_code=500,
            input_tokens=0,
            output_tokens=0,
            content=str(e),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8777, lifespan="on", log_level="debug")
