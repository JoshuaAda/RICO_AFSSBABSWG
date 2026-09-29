"""Standalone training function for Predictor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import numpy as np
from .training_utils import TrainLogger, create_optimizer, finalize_training, adaptive_clip_grad


@dataclass
class PredictorTrainConfig:
    """Training config for Predictor (stateless forward-pass training)."""

    N_epochs: int = 10000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_amsgrad: bool = True
    use_adaptive_clip_grad: bool = False
    print_frequency: int = 10
    log_values: bool = True
    use_fixed_val_batch: bool = True
    val_frequency: int = 50
    val_seed: int = 1234
    val_logtk_tolerance: float = 1e-6


if TYPE_CHECKING:
    from torch.nn import Module


def train_predictor(
    predictor: Module,
    nlp,
    config: PredictorTrainConfig,
    print_frequency: int | None = None,
    log_values: bool | None = None,
) -> TrainLogger:
    """Train a predictor model.

    Calls nlp.setup_for_batch(config.batch_size) internally before training.

    Args:
        predictor: Predictor-like nn.Module with forward(p_batch) -> z_batch.
        nlp: NLP problem instance.
        config: Typed training configuration.
        print_frequency: Optional override for how often to print progress.
        log_values: Optional override for whether to log training metrics.

    Returns:
        TrainLogger with training history.
    """
    # Configure NLP for this batch size
    nlp.setup_for_batch(config.batch_size,force_rebuild=False)

    # Unpack config
    N_epochs = config.N_epochs
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    lr = config.lr
    use_amsgrad = config.use_amsgrad
    use_adaptive_clip_grad = config.use_adaptive_clip_grad
    if print_frequency is None:
        print_frequency = config.print_frequency
    if log_values is None:
        log_values = config.log_values

    # Setup logger
    train_logger = TrainLogger(device=nlp.device)

    # Optional fixed validation batch (same size as train batch to avoid rebuilding batch funcs)
    p_val_batch = None
    if config.use_fixed_val_batch:
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(config.val_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.val_seed)
        p_val_batch = nlp.batch_gen_p(batch_size, offset=0.0)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    # Setup optimizer
    optim = create_optimizer(predictor.parameters(), lr=lr, weight_decay=weight_decay, use_amsgrad=use_amsgrad)

    # Training loop
    train_tic = time.perf_counter()
    for epoch in range(N_epochs):

        # Generate batch
        #p_batch=torch.Tensor(np.repeat(np.array([-2,2,1,-1]),256).reshape((4,256)).T).to("cuda").double()#np.array([-2,2,1,-1]).repmat(256,1)
        p_batch = nlp.batch_gen_p(batch_size, offset=0.0)

        # Forward pass
        optim.zero_grad(set_to_none=True)
        z_hat_batch = predictor(p_batch).to(p_batch.dtype)

        # Loss
        Tk_batch = predictor.Tk_batch_func(z_hat_batch, p_batch)
        #print(nlp.Fk_func_ca(np.array(z_hat_batch[0,:].cpu().detach()),np.array(p_batch[0,:].cpu().detach())))
        if torch.isnan(Tk_batch).any():
            raise RuntimeError("Tk_batch contains NaN")
        loss = torch.mean(torch.log(Tk_batch + predictor.offset**2))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if use_adaptive_clip_grad:
            adaptive_clip_grad(predictor.parameters())

        # Step
        optim.step()

        val_metrics = None
        # Log values
        if log_values:
            train_logger.log_value_buffered(epoch + 1, "epoch")
            train_logger.log_value_buffered(loss, "loss")

            if p_val_batch is not None and (
                (epoch + 1) % config.val_frequency == 0 or (epoch + 1) == N_epochs
            ):
                with torch.no_grad():
                    z_val_batch = predictor(p_val_batch).to(p_val_batch.dtype)
                    Tk_val_batch = predictor.Tk_batch_func(z_val_batch, p_val_batch)
                    logTk_val_batch = torch.log(Tk_val_batch + predictor.offset**2)

                    n_val = logTk_val_batch.numel()
                    k_90 = max(1, min(n_val, int(0.9 * n_val)))
                    k_95 = max(1, min(n_val, int(0.95 * n_val)))

                    train_logger.log_value_buffered(epoch + 1, "val_epoch")
                    val_logtk_median = logTk_val_batch.median()
                    val_logtk_90p = logTk_val_batch.kthvalue(k_90).values
                    val_logtk_95p = logTk_val_batch.kthvalue(k_95).values
                    val_conv_rate = (Tk_val_batch < config.val_logtk_tolerance).float().mean()

                    train_logger.log_value_buffered(val_logtk_median, "val_logTk_median")
                    train_logger.log_value_buffered(
                        val_logtk_90p, "val_logTk_90p"
                    )
                    train_logger.log_value_buffered(
                        val_logtk_95p, "val_logTk_95p"
                    )
                    train_logger.log_value_buffered(val_conv_rate, "val_convergence_rate")

                    val_metrics = {
                        "val_logTk_median": val_logtk_median,
                        "val_logTk_90p": val_logtk_90p,
                        "val_logTk_95p": val_logtk_95p,
                        "val_convergence_rate": val_conv_rate,
                    }

        # Print progress
        if (epoch + 1) % print_frequency == 0:
            print("Epoch {0}: Loss = {1:.3e}".format(epoch + 1, loss))
            if val_metrics is not None:
                print(
                    "           Val: median={0:.3e}, 90p={1:.3e}, 95p={2:.3e}, conv_rate={3:.3e}".format(
                        val_metrics["val_logTk_median"],
                        val_metrics["val_logTk_90p"],
                        val_metrics["val_logTk_95p"],
                        val_metrics["val_convergence_rate"],
                    )
                )
            print("-------------------------------")

    return finalize_training(train_logger, train_tic, epoch, log_values, p_final_batch=p_batch)
