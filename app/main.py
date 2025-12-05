import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import torch
import asyncio



# logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



# Global model variables
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForCausalLM] = None
pipe: Optional[pipeline] = None



@asynccontextmanager
async def llm_startup(app: FastAPI):
    global tokenizer, model, pipe
    model_path = os.getenv('MODEL_PATH', 'app/model_snapshot')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        logger.info(f'🚀 Initializing Transformers model from {model_path} on {device} device')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else None,
            local_files_only=True
        ).to(device)
        pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
        logger.info(f'✅ Model initialized successfully')
    except Exception as e:
        logger.error(f'❌ Error loading model: {e}')
    yield



# FastAPI
app = FastAPI(lifespan=llm_startup)

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'✅ Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'✅ Response status: {response.status_code}')
    return response



# LLM class
class LLM_communication(BaseModel):
    message: str
    temperature: float = 0.2
    max_tokens: int = 256



# FastAPI endpoints
@app.get('/')
async def healthcheck() -> JSONResponse:
    try:
        logger.info(f'✅ healthcheck')
        return JSONResponse(
            status_code=200,
            content={'status': '✅ LLM is healthy'}
        )
    except Exception as e:
        logger.error(f'❌ healthcheck failed: {e}')
        return JSONResponse(
            status_code=500,
            content={'❌ LLM is unhealthy': str(e)}
        )


@app.post('/generate')
async def llm_response(request: LLM_communication) -> JSONResponse:
    try:
        logger.info(f'✅ User prompt: {request.message}')
        result = await asyncio.to_thread(
            pipe,
            request.message,
            temperature=request.temperature,
            top_p=0.9,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            return_full_text=False
        )
        generated_text = result[0]['generated_text']
        logger.info(f'✅ LLM response: {generated_text}')
        return JSONResponse(
            status_code=200,
            content={'response': f'✅ LLM is healthy {generated_text}'}
        )
    except Exception as e:
        logger.error(f'❌ Error during generation: {e}')
        return JSONResponse(
            status_code=500,
            content={'❌ Error during generation': str(e)}
        )


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000
    )