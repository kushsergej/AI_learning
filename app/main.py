import os
import logging
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn



# logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



# FastAPI
app = FastAPI()

tokenizer = None
model = None
pipe = None

@app.on_event('startup')
def llm_startup():
    global tokenizer, model, pipe
    model_path = os.getenv('MODEL_PATH', 'app/model_snapshot')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        logger.info(f'🚀 Initializing Transformers model from {model_path} on {model} model')
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

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'✅ Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'✅ Response status: {response.status_code}')
    return response



# # LLM class
# class LLM_Request(BaseModel):
#     llm_prompt: str

# class LLM_Response(BaseModel):
#     llm_response: str



# FastAPI endpoints
@app.get('/')
def healthcheck():
    try:
        logger.info(f'✅ healthcheck')
        return JSONResponse(status_code=200, content={'status': '✅ LLM is healthy'})
    except Exception as e:
        logger.error(f'❌ healthcheck failed: {e}')
        return JSONResponse(status_code=500, content={'❌ LLM is unhealthy': str(e)})


@app.post('/generate')
def llm_response(prompt: str):
    if not model or not tokenizer:
        return JSONResponse(status_code=500, content={'❌ LLM is unhealthy': str(e)})
    try:
        logger.info(f'✅ User prompt: {prompt}')
        result = pipe(
            prompt,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=256,
            do_sample=True,
            return_full_text=False
        )
        generated_text = result[0]['generated_text']
        logger.info(f'✅ LLM response: {generated_text}')
        return JSONResponse(status_code=200, content={'response': f'✅ LLM is healthy {generated_text}'})
    except Exception as e:
        logger.error(f'❌ Error during generation: {e}')
        return JSONResponse(status_code=500, content={'❌ Error during generation': str(e)})


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000
    )