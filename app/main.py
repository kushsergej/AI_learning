import os
import logging
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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


# initialize pre-installed model
try:
    model_path = os.getenv('MODEL_PATH', 'app/model_snapshot')
    logger.info(f'✅ Initializing model from: {model_path}')
    engine = LLM(model=model_path, trust_remote_code=True)
    logger.info(f'✅ Model initialized successfully')
except Exception as e:
    engine = None
    print(f'❌ Error loading model: {e}')


# FastAPI
app = FastAPI()

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'✅ Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'✅ Response status: {response.status_code}')
    return response


# LLM class
class LLM_Request(BaseModel):
    llm_prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class LLM_Response(BaseModel):
    llm_response: str


# FastAPI endpoints
@app.get('/')
async def healthcheck() -> Response:
    try:
        logger.info(f'✅ healthcheck (GET)')
        return JSONResponse(status_code=200, content={'status': '✅ vLLM engine is healthy'})
    except Exception as e:
        logger.error(f'❌ Error during healthcheck (GET): {e}')
        return JSONResponse(status_code=500, content={'❌ vLLM is unhealthy (GET)': str(e)})


@app.post('/llm', response_model=LLM_Response)
async def llm_response(llm_request: LLM_Request):
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