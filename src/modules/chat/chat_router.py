from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .chat_service import ChatService
from src.shared.schemas import IncomingMessage

router = APIRouter(prefix='/chat', tags=['chat'])
chat_service = ChatService()

@router.post('/generate')
async def generate(message: IncomingMessage):
    return StreamingResponse(
        chat_service.generate_response(message),
        media_type='text/plain'
    )