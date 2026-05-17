from pydantic import BaseModel, Field
from src.shared.enums import MessageSender

class MessageItem(BaseModel):
  sender: MessageSender
  content: str

class IncomingMessage(BaseModel):
  old_messages: list[MessageItem]
  user_id: str
  content: str
