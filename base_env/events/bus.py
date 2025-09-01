# base_env/events/bus.py
# Descripción: Bus de eventos simple en memoria. Acumula eventos por step para logging/auditoría.
# Emitir con: events_bus.emit(kind="OPEN", ts=..., data={...})

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class Event:
    kind: str
    ts: int
    data: Dict[str, Any]

class SimpleEventBus:
    def __init__(self) -> None:
        self._buffer: List[Event] = []

    def emit(self, kind: str, ts: int, **data: Any) -> None:
        self._buffer.append(Event(kind=kind, ts=ts, data=data))

    def drain(self) -> List[Dict[str, Any]]:
        out = [ {"kind": e.kind, "ts": e.ts, **e.data} for e in self._buffer ]
        self._buffer.clear()
        return out
