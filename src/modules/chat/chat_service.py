import asyncio
from src.shared.schemas import IncomingMessage

class ChatService:
    async def generate_response(self, message: IncomingMessage):
        async for chunk in self._call_ai_model(message):
            yield chunk

    async def _call_ai_model(self, message: IncomingMessage):
        words = "Lorem ipsum doritatis sed? Expere suscipit aperiam soluta minima autem culpa ratione repellat ipsam quod quisquam. Placeat quisquam veniam dolorem rerum! Pariatur optio explicabo exercitationem numquam assumenda repellendus reprehenderit ratione eum nulla quas atque quaerat neque, maiores perspiciatis inventore molestias aspernatur quis facilis unde dolor autem repudiandae. Distinctio, aperiam est! Ducimus.".split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

