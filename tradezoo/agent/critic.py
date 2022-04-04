import torch
from .decision import DecisionBatch
from .observation import ObservationSeriesBatch


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observation_embedder = torch.nn.LSTM(
            input_size=6, hidden_size=64, batch_first=True
        )
        self.decision_embedder = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.LeakyReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 + 64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

    def evaluate(
        self, observations: ObservationSeriesBatch, decisions: DecisionBatch
    ) -> torch.Tensor:
        return self(observations.tensor, decisions.tensor)

    def forward(
        self, observations: torch.Tensor, decisions: torch.Tensor
    ) -> torch.Tensor:
        observation_embeddings = self.observation_embedder(observations)
        decision_embedding = self.decision_embedder(decisions)
        combined_embeddings = torch.cat(
            [observation_embeddings[:, -1], decision_embedding], dim=1
        )
        return self.head(combined_embeddings).squeeze(1)
