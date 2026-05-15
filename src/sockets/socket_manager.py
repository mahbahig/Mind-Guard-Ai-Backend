import socketio

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

@sio.event
async def connect(sid, environ):
    print(f"[socket] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[socket] Client disconnected: {sid}")

@sio.on('userMessage')
async def message_received(sid: str, data):
  await sio.emit("message_received", {"message": "Hi this is the bot response"}, to=sid)