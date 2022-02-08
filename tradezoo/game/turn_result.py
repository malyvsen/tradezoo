from dataclasses import dataclass

from tradezoo.agent import Action, Agent, Observation


@dataclass(frozen=True)
class TurnResult:
    agent: Agent
    observation: Observation
    action: Action
    reward: float
