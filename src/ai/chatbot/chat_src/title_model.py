# title_model.py — backward-compatible shim
from src.ai.chatbot.chat_src.model.attention import MultiHeadAttention
from src.ai.chatbot.chat_src.model.layers import FeedForward, EncoderLayer, DecoderLayer
from src.ai.chatbot.chat_src.model.blocks import PositionalEncoding, Encoder, Decoder
from src.ai.chatbot.chat_src.model.transformer import TitleTransformer
from src.ai.chatbot.chat_src.model.tokenizer import WordTokenizer

__all__ = ["TitleTransformer", "WordTokenizer", "MultiHeadAttention",
           "FeedForward", "EncoderLayer", "DecoderLayer",
           "PositionalEncoding", "Encoder", "Decoder"]
