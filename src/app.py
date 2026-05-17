from fastapi import FastAPI
from src.middlewares import register_all_middlewares
from src.modules import chat_router


def create_app() -> FastAPI:
  app = FastAPI(title='Mind Guard AI Backend')

  app.include_router(chat_router)

  register_all_middlewares(app)

  return app
