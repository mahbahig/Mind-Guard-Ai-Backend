from fastapi import FastAPI
import socketio
from src.middlewares import register_all_middlewares
from src.sockets import sio

def create_app() -> tuple[FastAPI, socketio.ASGIApp]:
    app = FastAPI(title="FastAPI + Socket.IO Demo")

    register_all_middlewares(app)

    full_app = socketio.ASGIApp(sio, other_asgi_app=app)

    return full_app