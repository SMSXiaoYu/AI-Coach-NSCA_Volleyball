import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from generation_hyde_rerank import retrieve_hyde_rerank, generate_response

app = FastAPI(title="AI 排球教练 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    question: str
    sources: list
    answer: str
    scores: list


@app.get("/")
def root():
    return {"message": "AI 排球教练 API", "status": "running"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        docs, sources, scores, hyde = retrieve_hyde_rerank(
            request.question, top_k=request.top_k
        )

        if not docs:
            return AnswerResponse(
                question=request.question,
                sources=[],
                answer="未找到相关内容",
                scores=[]
            )

        answer = generate_response(request.question, docs, sources)

        # 确保 scores 是 Python float
        scores_float = [float(s) for s in scores] if scores else []

        return AnswerResponse(
            question=request.question,
            sources=sources,
            answer=answer,
            scores=scores_float
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)