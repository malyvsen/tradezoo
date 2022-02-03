from dataclasses import dataclass

from tradezoo.agent import Action, Agent, State


@dataclass(frozen=True)
class TurnResult:
    agent: Agent
    state: State
    action: Action
    reward: float
