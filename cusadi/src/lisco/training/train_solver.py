"""Standalone training function for iterative solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import numpy as np
from .training_utils import TrainLogger, create_optimizer, finalize_training, adaptive_clip_grad


@dataclass
class SolverTrainConfig:
    """Training config for iterative solvers (Solver and subclasses)."""

    N_epochs: int = 1000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_amsgrad: bool = True
    resampling_strategy: Literal["all", "optimal_buffer","tree_trajectory"] = "all"
    Tk_lim: float = 1e-6
    n_steps_max_total: int | None = None
    beta: float = 0.05
    n_steps_max_no_improvement: int | None = None
    buffer_max_size: int = 1000
    p_buffer: float = 0.5
    p_fresh_min: float = 0.3
    use_adaptive_clip_grad: bool = False
    print_frequency: int = 100
    log_values: bool = True


if TYPE_CHECKING:
    from ...lisco.nn.solver import Solver


def train_solver(
    nlp,
    solver: Solver,
    config: SolverTrainConfig,
    print_frequency: int | None = None,
    log_values: bool | None = None,
) -> TrainLogger:
    """Train an iterative solver.

    Args:
        solver: Solver instance (or any subclass).
        config: Typed training configuration.
        print_frequency: Optional override for how often to print progress.
        log_values: Optional override for whether to log training metrics.

    Returns:
        TrainLogger with training history.
    """
    # Configure NLP for this batch size
    solver.nlp.setup_for_batch(config.batch_size, force_rebuild=False)

    # Unpack config
    N_epochs = config.N_epochs
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    lr = config.lr
    use_amsgrad = config.use_amsgrad
    resampling_strategy = config.resampling_strategy
    Tk_lim = config.Tk_lim
    beta = config.beta

    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta}")
    n_steps_max_total = config.n_steps_max_total
    n_steps_max_no_improvement = config.n_steps_max_no_improvement

    # Optimal buffer config
    buffer_max_size = config.buffer_max_size
    p_buffer = config.p_buffer
    p_fresh_min = config.p_fresh_min
    optimal_buffer = []
    use_adaptive_clip_grad = config.use_adaptive_clip_grad
    if print_frequency is None:
        print_frequency = config.print_frequency
    if log_values is None:
        log_values = config.log_values

    # Train logger
    train_logger = TrainLogger(device=solver.nlp.device)

    # Optimizer initialization
    optim = create_optimizer(solver.parameters(), lr=lr, weight_decay=weight_decay, use_amsgrad=use_amsgrad)

    # Start train timer
    train_tic = time.perf_counter()

    # Initial batch generation
    zk_batch, p_batch = solver.sample_z_p(batch_size)

    # Initial evaluations
    Tk_best = solver.nlp.Tk_batch_func(zk_batch, p_batch)
    Tk_best = Tk_best.detach().clone()
    n_steps_batch = torch.zeros(batch_size)
    n_steps_no_improv = torch.zeros(batch_size)

    for epoch in range(N_epochs):
        # if (epoch+1) % 5000 == 0 and (solver.nlp.eps>0.9*1e-3):
        #     new_eps = solver.nlp.eps*0.1
        #     solver.nlp.set_new_eps(new_eps)
        # Training step
        loss, zk_new, Tk_batch = solver.training_step(
            zk_batch,
            p_batch,
            optim,
            use_adaptive_clip_grad=use_adaptive_clip_grad,
        )

        # update step
        zk_batch = zk_new.detach().clone()

        with torch.no_grad():
            # update n_steps_batch and n_steps_no_improv
            n_steps_batch += 1
            # update Tk_best
            mask_best = Tk_batch < Tk_best
            Tk_best[mask_best] = Tk_batch[mask_best]
            n_steps_no_improv[~mask_best] += 1
            n_steps_no_improv[mask_best] = 0

            # Create boolean mask for resampling conditions
            mask_tol = Tk_batch < Tk_lim

            # Build max iteration mask based on strategy
            if n_steps_max_total is not None:
                mask_max_total = n_steps_batch >= n_steps_max_total
            else:
                mask_max_total = torch.zeros_like(mask_tol, dtype=torch.bool)

            if n_steps_max_no_improvement is not None:
                mask_max_no_improv = n_steps_no_improv >= n_steps_max_no_improvement
            else:
                mask_max_no_improv = torch.zeros_like(mask_tol, dtype=torch.bool)

            # Combine all masks with logical OR
            mask_resample = mask_tol | mask_max_total | mask_max_no_improv

            # Resample where necessary
            n_resample = torch.sum(mask_resample)
            n_resample_tol = torch.sum(mask_tol)
            n_resample_iter = n_resample - n_resample_tol
            # Add converged samples to buffer (before resampling)
            if resampling_strategy == "optimal_buffer" and mask_tol.any():
                for p_conv in p_batch[mask_tol]:
                    if len(optimal_buffer) < buffer_max_size:
                        optimal_buffer.append(p_conv.detach().clone())
                    else:  # FIFO replacement
                        optimal_buffer.pop(0)
                        optimal_buffer.append(p_conv.detach().clone())

            if n_resample > 0:
                if resampling_strategy == "all":
                    z_new, p_new = solver.sample_z_p(n_resample)
                    zk_batch[mask_resample] = z_new
                    p_batch[mask_resample] = p_new
                    Tk_best[mask_resample] = float('inf')
                    n_steps_batch[mask_resample] = 0
                    n_steps_no_improv[mask_resample] = 0
                elif resampling_strategy == "tree_trajectory":
                    # A sample that converged may continue down its current random tree.
                    # Samples that stopped because of iteration/no-improvement limits
                    # always restart from a fresh relaxed root.
                    mask_failed = (
                            (mask_max_total | mask_max_no_improv)
                            & ~mask_tol
                    )

                    restart_draw = (
                            torch.rand(
                                batch_size,
                                device=p_batch.device,
                            )
                            < beta
                    )

                    mask_root = mask_failed | (mask_tol & restart_draw)
                    mask_continue = mask_tol & ~restart_draw

                    # Build one child per continuing converged parent. zk_batch already
                    # contains zk_new.detach(), so successful children retain the complete
                    # converged parent primal-dual iterate.
                    continue_indices = torch.nonzero(
                        mask_continue,
                        as_tuple=False,
                    ).squeeze(1)

                    if continue_indices.numel() > 0:
                        p_children, has_free,branch_indices,branch_values = nlp.branch_random_child_batch(
                            p_batch[continue_indices]
                        )

                        child_indices = continue_indices[has_free]
                        p_batch[child_indices] = p_children[has_free]

                        # Fully fixed parents cannot be branched further and therefore
                        # become fresh relaxed roots regardless of beta.
                        # Project the newly fixed primal binary variable onto its child bound.
                        #selected_branch_indices = branch_indices[has_free]
                        #selected_branch_values = branch_values[has_free].to(
                        #    dtype=zk_batch.dtype
                        #)

                        #binary_primal_offset = (
                        #        nlp.N * nlp.nx
                        #        + nlp.N * nlp.nmpc.nu
                        #)

                        #zk_batch[
                        #    child_indices,
                        #    binary_primal_offset + selected_branch_indices,
                        #] = selected_branch_values

                        # Fully fixed parents become fresh relaxed roots.
                        no_free_indices = continue_indices[~has_free]

                        mask_root[no_free_indices] = True
                        mask_continue[no_free_indices] = False

                    # Generate a complete batch of relaxed roots and predictor
                    # initializations. Only rows selected by mask_root are copied.
                    # This keeps the predictor evaluation at its configured batch size.
                    if mask_root.any():
                        p_root_all = nlp.batch_gen_p(
                            batch_size,
                            offset=0.0,
                        )

                        if solver.predictor is not None:
                            zk_root_all = solver.predictor.predict(
                                p_root_all
                            ).to(nlp.dtype)
                        else:
                            zk_root_all = nlp.batch_gen_z(batch_size)

                        p_batch[mask_root] = p_root_all[mask_root]
                        zk_batch[mask_root] = zk_root_all[mask_root]

                    # Both a child and a root constitute a new optimization problem.
                    mask_new_problem = mask_continue | mask_root

                    Tk_best[mask_new_problem] = float("inf")
                    n_steps_batch[mask_new_problem] = 0
                    n_steps_no_improv[mask_new_problem] = 0
                elif resampling_strategy == "optimal_buffer":
                    n_resample_int = int(n_resample.item())
                    # Determine fresh vs buffer split
                    n_fresh = max(int(n_resample_int * p_fresh_min), 1)
                    n_from_buffer = 0
                    if len(optimal_buffer) > 0:
                        n_from_buffer = int((n_resample_int - n_fresh) * p_buffer)
                    n_fresh = n_resample_int - n_from_buffer

                    # Sample fresh
                    z_new, p_new = solver.sample_z_p(n_fresh)

                    # Sample from buffer using farthest-point selection
                    if n_from_buffer > 0:
                        p_buf = torch.stack(optimal_buffer)  # [buf_size, p_dim]
                        p_buf_selected = []
                        for _ in range(n_from_buffer):
                            # Distance to current batch + already selected
                            p_current = torch.cat([p_batch[~mask_resample], p_new] +
                                                  ([torch.stack(p_buf_selected)] if p_buf_selected else []), dim=0)
                            dists = torch.cdist(p_buf, p_current).min(dim=1).values
                            idx = dists.argmax().item()
                            p_buf_selected.append(p_buf[idx])
                        p_from_buf = torch.stack(p_buf_selected)
                        z_from_buf = solver.predictor.predict(p_from_buf).to(solver.nlp.dtype) if solver.predictor else solver.nlp.batch_gen_z(n_from_buffer)
                        z_new = torch.cat([z_new, z_from_buf], dim=0)
                        p_new = torch.cat([p_new, p_from_buf], dim=0)

                    zk_batch[mask_resample] = z_new
                    p_batch[mask_resample] = p_new
                    Tk_best[mask_resample] = float('inf')
                    n_steps_batch[mask_resample] = 0
                    n_steps_no_improv[mask_resample] = 0
                else:
                    raise NotImplementedError("Resampling strategy not implemented yet.")

            # Print
            if (epoch+1) % print_frequency == 0:
                print("Epoch {0}: Loss = {1:.3e}, Tk_min = {2:.3e}, N_tol={3:}".format(epoch+1, loss, Tk_batch.min().item(), n_resample_tol.item()))
                print("-------------------------------")

            # Log values
            if log_values:
                train_logger.log_value_buffered(epoch+1, "epoch")
                train_logger.log_value_buffered(loss, "loss")
                train_logger.log_value_buffered(n_resample, "n_resample")
                train_logger.log_value_buffered(n_resample_tol, "n_resample_tol")
                train_logger.log_value_buffered(n_resample_iter, "n_resample_iter")
                train_logger.log_value_buffered(len(optimal_buffer), "buffer_size")

    return finalize_training(train_logger, train_tic, epoch, log_values, p_final_batch=p_batch)
