from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable


@dataclass(frozen=True)
class SelectionState:
    original_sample_id: str | None = None
    real_anomaly_id: str | None = None
    synthetic_anomaly_id: str | None = None
    hybrid_sample_id: str | None = None
    placement_id: str | None = None
    source: str | None = None


class SelectionController:
    """Shares entity selection between tabs without depending on Tkinter."""

    def __init__(self) -> None:
        self.state = SelectionState()
        self._listeners: list[Callable[[SelectionState], None]] = []

    def subscribe(self, listener: Callable[[SelectionState], None]) -> None:
        self._listeners.append(listener)

    def update(self, *, source: str | None = None, **changes) -> SelectionState:
        unknown = set(changes) - set(SelectionState.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown selection fields: {sorted(unknown)}")
        self.state = replace(self.state, source=source, **changes)
        for listener in tuple(self._listeners):
            listener(self.state)
        return self.state

    def clear(self, *, source: str | None = None) -> SelectionState:
        self.state = SelectionState(source=source)
        for listener in tuple(self._listeners):
            listener(self.state)
        return self.state
