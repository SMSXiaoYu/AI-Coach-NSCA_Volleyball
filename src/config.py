import os
from dotenv import load_dotenv

load_dotenv()

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 数据路径
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
CHROMA_DIR = "chroma_db"

# 所有已处理的文本文件（Phase 1 会遍历这个列表）
PROCESSED_FILES = [
    "data/processed/nsca_guide.txt",
]

# 大模型配置
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4