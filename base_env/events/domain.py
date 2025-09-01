
"""
base_env/events/domain.py
Descripción: Eventos de dominio (order_opened/closed, sl/tp, breaker, data_quality) + bus simple para recolectarlos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass(frozen=True)
class DomainEvent:
    ts: int
    type: str
    data: Dict[str, Any]

    @staticmethod
    def ts_now(evt_type: str, data: Dict[str, Any]) -> "DomainEvent":
        return DomainEvent(ts=int(time.time() * 1000), type=evt_type, data=data)


class EventBus:
    def __init__(self) -> None:
        self._events: List[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self._events.append(event)

    def drain(self) -> List[DomainEvent]:
        out = self._events[:]
        self._events.clear()
        return out
