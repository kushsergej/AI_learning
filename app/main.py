import os
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



# load pre-installed model
try:
    model_path = os.getenv('MODEL_PATH', './model_cache')
    logger.info(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
    logger.info(f'✅ Model loaded successfully')
except Exception as e:
    pipe = None
    logger.info(f'❌ Error loading model: {e}')



# FastAPI
app = FastAPI()

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'Response status: {response.status_code}')
    return response



# LLM class
class LLM_Request(BaseModel):
    llm_prompt: str

class LLM_Response(BaseModel):
    llm_response: str



# FastAPI endpoints
@app.get('/')
async def healthcheck() -> Response:
    if pipe:
        return JSONResponse(status_code=200, content={'status': 'Healthy'})
    return JSONResponse(status_code=500, content={'error': 'Model not loaded'})



@app.post('/llm', response_model=LLM_Response)
async def llm_response(llm_request: LLM_Request):
    if not pipe:
        return JSONResponse(status_code=500, content={'error': 'Model not loaded'})
    try:
        logger.info(f'Initial prompt: {llm_request.llm_prompt}')

        result = pipe(
            llm_request.llm_prompt,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=256,
            do_sample=True,
            return_full_text=False
        )
        return LLM_Response(llm_response=f'{result[0]['generated_text']} [from {model}]')
    except Exception as e:
        return JSONResponse(status_code=500, content={'err message': str(e)})



if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000
        )