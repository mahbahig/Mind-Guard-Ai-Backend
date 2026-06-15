import asyncio
from src.shared.schemas import IncomingMessage
from src.ai.chatbot import run_chat

class ChatService:
    async def generate_response(self, message: IncomingMessage):
        async for chunk in self._call_ai_model(message):
            yield chunk

    async def _call_ai_model(self, message: IncomingMessage):
        print(f"User Message: {message.content}")
        response = run_chat(
            query           = message.content,
            history         = [],
            llm_backend     = "vertex_tuned",
            history_summary = "",
        )
        print(f"AI Model Response: {response}")
        response_message = response[0]
        for word in response_message.split():
            yield word + " "
            await asyncio.sleep(0.01)

