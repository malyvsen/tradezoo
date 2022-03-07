from dataclasses import dataclass


@dataclass(frozen=True)
class TrainResult:
    actor_loss: float
    critic_loss: float
