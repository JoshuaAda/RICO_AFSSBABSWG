"""Common utilities for CSTR training scripts.

This module provides shared functionality for solver and predictor training,
including NLP setup, configuration serialization, and metrics computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import dataclasses

import numpy as np
import torch

from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import NMPCParams, CSTRNMPC
from src.nlp_handler import NLPHandler, NLPConfig


def project_root() -> Path:
    """Return the project root directory.
    
    Assumes common.py is located at cusadi/src/common.py.
    Returns the path to the project root (cusadi/).
    
    Returns:
        Path: Absolute path to the project root.
    """
    # common.py is at cusadi/src/common.py, so go up 2 levels to cusadi/
    return Path(__file__).parent.parent.resolve()


def dict_config_to_json(config: Any) -> Dict[str, Any]:
    """Convert a configuration object to a JSON-serializable dictionary.
    
    Handles dataclasses and regular dicts, converting nested structures
    recursively and handling special types like numpy arrays, tensors, etc.
    
    Args:
        config: A dataclass instance, dict, or other serializable object.
        
    Returns:
        Dict: A JSON-serializable dictionary representation of the config.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        # Convert dataclass to dict, recursively handling nested dataclasses
        result = {}
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            result[field.name] = dict_config_to_json(value)
        return result
    
    elif isinstance(config, dict):
        # Recursively convert dict values
        result = {}
        for key, value in config.items():
            result[key] = dict_config_to_json(value)
        return result
    
    elif isinstance(config, (list, tuple)):
        # Recursively convert list/tuple elements
        return [dict_config_to_json(item) for item in config]
    
    elif isinstance(config, np.ndarray):
        # Convert numpy arrays to lists
        return config.tolist()
    
    elif isinstance(config, torch.Tensor):
        # Convert PyTorch tensors to lists
        return config.detach().cpu().numpy().tolist()
    
    elif isinstance(config, (np.integer, np.floating)):
        # Convert numpy scalars to Python types
        return config.item()
    
    elif isinstance(config, (float, int, str, bool, type(None))):
        # Return primitives as-is
        return config
    
    else:
        # For unknown types, try to convert to string
        try:
            return str(config)
        except Exception as e:
            raise TypeError(f"Cannot serialize {type(config).__name__}: {config}") from e


def get_metrics(values: torch.Tensor | np.ndarray | list) -> Dict[str, float]:
    """Compute summary statistics from a batch of values.
    
    Computes mean, std, min, max, median, and percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99).
    
    Args:
        values: Tensor, array, or list of numeric values (1D).
        
    Returns:
        Dict: Dictionary with keys like 'mean', 'std', 'min', 'max', 'median',
              'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'.
    """
    # Convert to numpy for statistical computation
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        values_np = values
    else:
        values_np = np.array(values)
    
    # Flatten if needed
    values_flat = values_np.flatten()
    
    # Compute statistics
    metrics = {
        'mean': float(np.mean(values_flat)),
        'std': float(np.std(values_flat)),
        'min': float(np.min(values_flat)),
        'max': float(np.max(values_flat)),
        'median': float(np.median(values_flat)),
    }
    
    # Add percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 80, 85, 90, 95, 99]
    for p in percentiles:
        key = f'p{p}'
        metrics[key] = float(np.percentile(values_flat, p))
    
    return metrics


def setup_nlp_from_params(
    model_params: Dict[str, Any] | CSTRParams,
    nmpc_params: Dict[str, Any] | NMPCParams,
    nlp_config: Dict[str, Any] | NLPConfig,
) -> Tuple[CSTRNMPC, NLPHandler]:
    """Set up NMPC and NLPHandler from parameter dictionaries or objects.
    
    This function orchestrates the creation of:
    1. A CSTRParams object (if not already provided)
    2. An NMPCParams object (if not already provided)
    3. A CSTRNMPC instance (the OCP formulation)
    4. An NLPHandler instance (the NLP interface with CasADi functions)
    
    Args:
        model_params: Dictionary or CSTRParams object with CSTR model parameters.
            If dict, must contain keys like 'nr', 'dt', etc. matching CSTRParams fields.
        nmpc_params: Dictionary or NMPCParams object with NMPC parameters.
            If dict, must contain keys like 'N', 'w_R', 'w_S', etc. matching NMPCParams fields.
        nlp_config: Dictionary or NLPConfig object with NLP configuration.
            If dict, must contain keys like 'device', 'dtype', etc. matching NLPConfig fields.
            
    Returns:
        Tuple[CSTRNMPC, NLPHandler]: 
            - nmpc: The CSTRNMPC instance
            - nlp: The NLPHandler instance wrapping the NMPC
    """
    # Convert dict inputs to dataclass instances if needed
    if isinstance(model_params, dict):
        model_params = CSTRParams(**model_params)
    
    if isinstance(nmpc_params, dict):
        nmpc_params = NMPCParams(**nmpc_params)
    
    if isinstance(nlp_config, dict):
        nlp_config = NLPConfig(**nlp_config)
    
    # Create the NMPC problem
    nmpc = CSTRNMPC(model_params, nmpc_params)
    
    # Create the NLPHandler for optimization interface
    nlp = NLPHandler(
        nmpc,
        device=nlp_config.device,
        dtype=nlp_config.dtype,
        eps=nlp_config.eps,
        sigma=nlp_config.sigma,
        n_threads=nlp_config.n_threads,
        distributed_KKT_functions=nlp_config.distributed_KKT_functions,
    )
    
    return nmpc, nlp


def quick_eval_solver(
    solver: torch.nn.Module,
    nlp: NLPHandler,
    n_eval: int = 100,
    max_iter: int = 200,
) -> Dict[str, Any]:
    """Evaluate a trained solver on test problems.
    
    Performs a quick evaluation of the solver on generated test cases,
    computing metrics like KKT residual, optimality, and convergence statistics.
    
    Args:
        solver: A trained Solver or SolverDecomposed instance.
        nlp: The NLPHandler instance.
        n_eval: Number of test problems to solve.
        max_iter: Maximum iterations for each problem.
        
    Returns:
        Dict: Dictionary containing evaluation metrics (KKT residual, optimality info, etc.).
    """
    # Generate random test parameters
    p_eval = nlp.batch_gen_p(n_eval)
    
    # Solve batch
    z_solved, _, _ = solver.solve_batch(
        p_batch=p_eval,
        max_iter=max_iter,
        alpha=1.0,
        predictor=getattr(solver, 'predictor', None),
        update_mode='line_search',
        return_trajectory=False,
    )
    
    # Evaluate KKT conditions
    Tk_eval = nlp.Tk_batch_func(z_solved, p_eval)
    KKT_eval = nlp.KKT_batch_func(z_solved, p_eval)
    KKT_inf_norm = KKT_eval.norm(p=float('inf'), dim=1)
    
    # Compute metrics
    metrics = {
        'kkt_residual': get_metrics(KKT_inf_norm),
        'optimality': get_metrics(Tk_eval),
        'n_eval': n_eval,
        'max_iter': max_iter,
    }
    
    return metrics


def plot_predictor_results(
    train_logger,
    save_dir: Path,
    title: str = "Predictor Training History",
) -> None:
    """Save predictor training plots from TrainLogger history.
    
    Args:
        train_logger: TrainLogger instance with training history.
        save_dir: Directory to save plot files.
        title: Title for the plots.
    """
    import matplotlib.pyplot as plt
    
    plot_specs = [
        ("loss", "Predictor Training Loss", save_dir / "loss.png"),
    ]
    
    for key, fig_title, save_path in plot_specs:
        if key not in train_logger.history:
            continue
            
        try:
            fig, _ = train_logger.plot_history(
                keys_to_plot=[key],
                save_path=save_path,
                x_key="epoch",
                title=fig_title,
            )
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not plot {key}: {e}")
    
    print(f"Training plots saved to: {save_dir}")
