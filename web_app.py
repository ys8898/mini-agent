"""独立的 FastAPI 入口；复用现有 Agent 与模型请求函数。"""

from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent import run_agent
from request_model import request_model


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class HealthResponse(BaseModel):
    status: str


class Session:
    def __init__(self):
        self.messages: list[dict] = []
        self.lock = Lock()


app = FastAPI(title="Mini Tool Calling Agent")
_sessions: dict[str, Session] = {}
_registry_lock = Lock()


def _get_session(session_id: str) -> Session:
    # 两个并发请求若同时首次使用同一 ID，只能创建一个共享的 Session。
    with _registry_lock:
        if session_id not in _sessions:
            _sessions[session_id] = Session()
        return _sessions[session_id]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    message = body.message.strip()
    if not message or (body.session_id is not None and not body.session_id.strip()):
        raise HTTPException(status_code=422, detail="消息和会话 ID 不能只包含空白字符")

    session_id = body.session_id or uuid4().hex
    session = _get_session(session_id)
    try:
        # HTTP 端点是同步函数；同一会话的锁覆盖整个 Agent 轮次，保护消息顺序。
        with session.lock:
            answer = run_agent(message, request_model, messages=session.messages)
    except (RuntimeError, ValueError):
        raise HTTPException(status_code=502, detail="Agent 或模型请求失败") from None
    return ChatResponse(session_id=session_id, answer=answer)
