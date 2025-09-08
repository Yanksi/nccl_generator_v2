from typing import List, Dict, Type, Union, Optional
from dataclasses import dataclass
from __future__ import annotations

@dataclass
class GoalRank:
    id: int

@dataclass
class GoalPrimitive:
    dependencies: List[GoalPrimitive] = None

@dataclass
class GoalCalc(GoalPrimitive):
    duration: int = 0

@dataclass
class GoalSend(GoalPrimitive):
    size: int = 0
    peer_rank: GoalRank = None
    tag: int = -1
    send: bool = True

@dataclass
class GoalRecv(GoalSend):
    send: bool = False