from fastapi import FastAPI
from src.middlewares.cors_middleware import register_cors_middleware

def register_all_middlewares(app: FastAPI):
  register_cors_middleware(app)