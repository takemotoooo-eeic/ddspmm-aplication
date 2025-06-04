from dotenv import load_dotenv

env_path = ".env"
load_dotenv(env_path)

from fastapi import FastAPI

from api.controllers.backend_api.router import backend_api_router

app = FastAPI()


backend_api = FastAPI(title="backend_api")
backend_api.include_router(backend_api_router)


@app.get("/health", status_code=200)
def healthcheck():
    return {"status": "OK"}


app.mount("/backend-api", backend_api)
