from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


_REQUIRED_KEYS = {
    "features",
    "teacher_scores",
    "candidate_indices",
    "instance_id",
}


class BranchingDataset(Dataset):
    """Node-level branching examples produced by offline strong branching."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.examples = torch.load(
            self.path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(self.examples, list):
            raise TypeError("Branching dataset must contain a list of examples")
        for index, example in enumerate(self.examples):
            missing = _REQUIRED_KEYS.difference(example)
            if missing:
                raise ValueError(
                    f"Example {index} is missing required keys: {sorted(missing)}"
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = dict(self.examples[index])
        example["features"] = torch.as_tensor(
            example["features"], dtype=torch.float32
        )
        example["teacher_scores"] = torch.as_tensor(
            example["teacher_scores"], dtype=torch.float32
        )
        example["candidate_indices"] = torch.as_tensor(
            example["candidate_indices"], dtype=torch.long
        )
        return example


class BranchingExampleWriter:
    """Accumulates offline strong-branching labels in the training format."""

    def __init__(self):
        self.examples: list[dict[str, Any]] = []

    def add(
        self,
        *,
        features,
        teacher_scores,
        candidate_indices,
        instance_id: int | str,
        x0=None,
        node_depth: int | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        teacher_scores = torch.as_tensor(
            teacher_scores, dtype=torch.float32
        ).reshape(-1).cpu()
        candidate_indices = torch.as_tensor(
            candidate_indices, dtype=torch.long
        ).reshape(-1).cpu()

        n_candidates = features.shape[0]
        if teacher_scores.shape[0] != n_candidates:
            raise ValueError("teacher_scores and features have different lengths")
        if candidate_indices.shape[0] != n_candidates:
            raise ValueError("candidate_indices and features have different lengths")

        example: dict[str, Any] = {
            "features": features,
            "teacher_scores": teacher_scores,
            "candidate_indices": candidate_indices,
            "instance_id": instance_id,
        }
        if x0 is not None:
            example["x0"] = torch.as_tensor(x0, dtype=torch.float32).reshape(-1)
        if node_depth is not None:
            example["node_depth"] = int(node_depth)
        if feature_names is not None:
            example["feature_names"] = list(feature_names)
        self.examples.append(example)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.examples, path)