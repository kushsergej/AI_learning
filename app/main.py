import os
import logging
import asyncio
from fastapi import FastAPI, Request, Response
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

pipe = None

@app.on_event('startup')
async def init_llm():
    global pipe
    model_path = os.getenv('MODEL_PATH')
    try:
        logger.info(f'🚀 Initializing Transformers model from: {model_path}')
        tokenizer = await asyncio.to_thread(lambda: AutoTokenizer.from_pretrained(model_path, use_fast=True))
        model = await asyncio.to_thread(lambda: AutoModelForCausalLM.from_pretrained(model_path))
        pipe = await asyncio.to_thread(lambda: pipeline(task='text-generation', model=model, tokenizer=tokenizer))
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


# # FastAPI endpoints
# @app.get('/')
# async def healthcheck() -> Response:
#     try:
#         logger.info(f'✅ healthcheck (GET)')
#         return JSONResponse(status_code=200, content={'status': '✅ vLLM engine is healthy'})
#     except Exception as e:
#         logger.error(f'❌ Error during healthcheck (GET): {e}')
#         return JSONResponse(status_code=500, content={'❌ vLLM is unhealthy (GET)': str(e)})


# @app.post('/llm', response_model=LLM_Response)
# async def llm_response(llm_request: LLM_Request):
    if not engine:
        return JSONResponse(status_code=500, content={'error': '❌ vLLM engine is unhealthy (POST)'})
    try:
        logger.info(f'✅ User prompt: {llm_request.llm_prompt}')

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=256
        )
        result = engine.generate([llm_request.llm_prompt], sampling_params)

        return LLM_Response(llm_response=f'{result[0].outputs[0].text} [from vLLM]')
    except Exception as e:
        logger.error(f'❌ Error during generation: {e}')
        return JSONResponse(status_code=500, content={'❌ Error during generation': str(e)})


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000
        )