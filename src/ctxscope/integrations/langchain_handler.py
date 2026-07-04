"""CtxScopeCallbackHandler: attach to any LangChain/LangGraph runnable."""

from __future__ import annotations
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
except ImportError as e:
    raise ImportError("pip install 'ctxscope[langchain]'") from e


def _to_dict(msg: "BaseMessage") -> dict:
    role = {"human": "user", "ai": "assistant", "system": "system",
            "tool": "user"}.get(msg.type, msg.type)
    content = msg.content
    # LangChain ToolMessage -> represent as a tool_result block so
    # classification works identically to the raw-SDK path.
    if msg.type == "tool":
        content = [{"type": "tool_result",
                    "tool_use_id": getattr(msg, "tool_call_id", "?"),
                    "content": msg.content}]
    return {"role": role, "content": content}


class CtxScopeCallbackHandler(BaseCallbackHandler):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_chat_model_start(self, serialized: dict,
                             messages: list[list["BaseMessage"]],
                             **kwargs: Any) -> None:
        model = (serialized or {}).get("kwargs", {}).get("model")
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        for batch in messages:
            self.profiler.record_turn(
                [_to_dict(m) for m in batch],
                model=model,
                meta={"node": node} if node else {},
            )
