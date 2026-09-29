from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from branching.branching_dataset import BranchingDataset
from branching.branching_policy import BranchingPolicy, BranchingPolicyConfig


def split_by_instance(examples, validation_fraction: float, seed: int):
    instance_ids = sorted({str(example["instance_id"]) for example in examples})
    generator = random.Random(seed)
    generator.shuffle(instance_ids)

    n_validation = max(1, round(len(instance_ids) * validation_fraction))
    validation_ids = set(instance_ids[:n_validation])
    train_indices = [
        index
        for index, example in enumerate(examples)
        if str(example["instance_id"]) not in validation_ids
    ]
    validation_indices = [
        index
        for index, example in enumerate(examples)
        if str(example["instance_id"]) in validation_ids
    ]
    if not train_indices or not validation_indices:
        raise ValueError("Dataset needs at least two distinct instance_id values")
    return train_indices, validation_indices


def iter_batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def collate_nodes(dataset, indices, device):
    examples = [dataset[index] for index in indices]
    max_candidates = max(example["features"].shape[0] for example in examples)
    feature_dim = examples[0]["features"].shape[1]
    batch_size = len(examples)

    features = torch.zeros(
        batch_size,
        max_candidates,
        feature_dim,
        dtype=torch.float32,
        device=device,
    )
    teacher_scores = torch.full(
        (batch_size, max_candidates),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    candidate_indices = torch.full(
        (batch_size, max_candidates),
        -1,
        dtype=torch.long,
        device=device,
    )

    for row, example in enumerate(examples):
        count = example["features"].shape[0]
        features[row, :count] = example["features"].to(device)
        teacher_scores[row, :count] = example["teacher_scores"].to(device)
        candidate_indices[row, :count] = example["candidate_indices"].to(device)

    valid = torch.isfinite(teacher_scores)
    usable_rows = valid.sum(dim=1) >= 2
    return (
        features[usable_rows],
        teacher_scores[usable_rows],
        candidate_indices[usable_rows],
        valid[usable_rows],
    )


def node_ranking_loss(
    logits,
    teacher_scores,
    valid,
    temperature,
    loss_type,
):
    masked_logits = logits.masked_fill(~valid, -torch.inf)

    if loss_type == "hard":
        targets = teacher_scores.masked_fill(~valid, -torch.inf).argmax(dim=1)
        return torch.nn.functional.cross_entropy(
            masked_logits,
            targets,
            reduction="none",
        )

    valid_float = valid.to(teacher_scores.dtype)
    counts = valid_float.sum(dim=1, keepdim=True)
    finite_scores = teacher_scores.masked_fill(~valid, 0.0)
    means = finite_scores.sum(dim=1, keepdim=True) / counts
    centered = (finite_scores - means).masked_fill(~valid, 0.0)
    standard_deviations = (
        centered.square().sum(dim=1, keepdim=True) / counts
    ).sqrt().clamp_min(1e-6)
    normalized_scores = (teacher_scores - means) / standard_deviations
    target_logits = (normalized_scores / temperature).masked_fill(
        ~valid,
        -torch.inf,
    )

    target = torch.softmax(target_logits, dim=1)
    log_target = torch.where(
        valid,
        target.clamp_min(1e-12).log(),
        torch.zeros_like(target),
    )
    log_prediction = torch.log_softmax(masked_logits, dim=1)
    return (
        target
        * (log_target - log_prediction.masked_fill(~valid, 0.0))
    ).sum(dim=1)


def batch_metrics(
    logits,
    teacher_scores,
    candidate_indices,
    valid,
    variables_per_stage,
):
    masked_logits = logits.masked_fill(~valid, -torch.inf)
    masked_teacher = teacher_scores.masked_fill(~valid, -torch.inf)
    predicted_positions = masked_logits.argmax(dim=1)
    teacher_positions = masked_teacher.argmax(dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device)

    predicted_variables = candidate_indices[rows, predicted_positions]
    teacher_variables = candidate_indices[rows, teacher_positions]
    variable_correct = predicted_variables == teacher_variables
    stage_correct = (
        predicted_variables // variables_per_stage
        == teacher_variables // variables_per_stage
    )
    return int(variable_correct.sum()), int(stage_correct.sum()), logits.shape[0]


@torch.no_grad()
def evaluate(
    policy,
    dataset,
    indices,
    device,
    batch_size,
    temperature,
    loss_type,
    variables_per_stage,
):
    policy.eval()
    loss_sum = 0.0
    variable_correct = 0
    stage_correct = 0
    evaluated = 0

    for batch_indices in iter_batches(indices, batch_size):
        features, scores, candidates, valid = collate_nodes(
            dataset,
            batch_indices,
            device,
        )
        if features.shape[0] == 0:
            continue

        logits = policy(features)
        losses = node_ranking_loss(
            logits,
            scores,
            valid,
            temperature,
            loss_type,
        )
        batch_variable, batch_stage, batch_count = batch_metrics(
            logits,
            scores,
            candidates,
            valid,
            variables_per_stage,
        )
        loss_sum += float(losses.sum())
        variable_correct += batch_variable
        stage_correct += batch_stage
        evaluated += batch_count

    denominator = max(evaluated, 1)
    return (
        loss_sum / denominator,
        variable_correct / denominator,
        stage_correct / denominator,
    )


def train(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.variables_per_stage < 1:
        raise ValueError("variables-per-stage must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = BranchingDataset(args.dataset)
    train_indices, validation_indices = split_by_instance(
        dataset.examples,
        args.validation_fraction,
        args.seed,
    )

    train_features = torch.cat(
        [dataset[index]["features"] for index in train_indices],
        dim=0,
    )
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-8)

    config = BranchingPolicyConfig(
        input_dim=train_features.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
    )
    device = torch.device(args.device)
    policy = BranchingPolicy(config).to(device)
    policy.set_normalization(feature_mean, feature_std)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    random_generator = random.Random(args.seed)
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        shuffled = list(train_indices)
        random_generator.shuffle(shuffled)

        training_loss_sum = 0.0
        training_variable_correct = 0
        training_stage_correct = 0
        training_count = 0

        for batch_indices in iter_batches(shuffled, args.batch_size):
            features, scores, candidates, valid = collate_nodes(
                dataset,
                batch_indices,
                device,
            )
            if features.shape[0] == 0:
                continue

            logits = policy(features)
            losses = node_ranking_loss(
                logits,
                scores,
                valid,
                args.temperature,
                args.loss,
            )
            loss = losses.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_variable, batch_stage, batch_count = batch_metrics(
                logits.detach(),
                scores,
                candidates,
                valid,
                args.variables_per_stage,
            )
            training_loss_sum += float(losses.detach().sum())
            training_variable_correct += batch_variable
            training_stage_correct += batch_stage
            training_count += batch_count

        validation_loss, validation_variable, validation_stage = evaluate(
            policy,
            dataset,
            validation_indices,
            device,
            args.batch_size,
            args.temperature,
            args.loss,
            args.variables_per_stage,
        )
        denominator = max(training_count, 1)
        training_loss = training_loss_sum / denominator
        training_variable = training_variable_correct / denominator
        training_stage = training_stage_correct / denominator

        print(
            f"epoch={epoch:04d} "
            f"train_loss={training_loss:.6f} "
            f"train_top1={training_variable:.4f} "
            f"train_stage_top1={training_stage:.4f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_top1={validation_variable:.4f} "
            f"validation_stage_top1={validation_stage:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            feature_names = dataset.examples[0].get("feature_names", [])
            policy.save(args.output, feature_names=feature_names)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument(
        "--activation",
        choices=("relu", "gelu", "tanh"),
        default="relu",
    )
    parser.add_argument(
        "--loss",
        choices=("normalized-soft", "hard"),
        default="normalized-soft",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--variables-per-stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
