from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages:   Annotated[list, add_messages]
    next:       str
    role:       str
    username:   str
    turn_count: int   # ← tracks supervisor re-routing; breaks infinite loops