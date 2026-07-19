"""
EmbodimentAdapter — the cross-robot extension point.

An adapter turns a machine-agnostic EEFrame (6-DOF pose + gripper, and later a
hand skeleton) into joint targets for ONE robot. Adding a new arm or a
dexterous hand = one new adapter; the perception/transport/controller stack is
untouched. This is what makes captured motion reusable beyond SO-101.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from eeframe import EEFrame


class EmbodimentAdapter(ABC):
    name: str = "abstract"

    @property
    @abstractmethod
    def ids(self) -> list[int]:
        """Actuator ids this adapter drives, in canonical order."""

    @abstractmethod
    def home_ticks(self) -> dict[int, int]:
        """Safe rest pose in servo ticks."""

    @abstractmethod
    def retarget(self, frame: EEFrame) -> dict[int, int]:
        """Map an EEFrame to per-actuator goal ticks (pre-safety-clamp)."""

    def capabilities(self) -> dict:
        """Machine-readable descriptor; lets a planner reason about reach/DOF."""
        return {"name": self.name, "dof": len(self.ids)}
