"""Solver training script for CSTR Cascade Economic NMPC.

Trains either a monolithic Solver or a windowed SolverDecomposed, with optional
predictor initialization, using a unified dict-based config structure similar to
`train_pred_cstr.py`.

Usage:
    python src/experiments/train_solver_cstr.py
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from optimization import Linearized_Four_Tank,Four_Tank
import torch

try:
    from .common import (
        setup_nlp_from_params,
        dict_config_to_json,
        get_metrics,
        project_root,
        # quick_eval_solver,
    )
except ImportError:
    # Fallback: manually load common.py from the same directory
    import sys
    import importlib.util
    from pathlib import Path
    
    common_path = Path(__file__).parent / "common.py"
    spec = importlib.util.spec_from_file_location("common", common_path)
    common_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common_module)
    
    setup_nlp_from_params = common_module.setup_nlp_from_params
    dict_config_to_json = common_module.dict_config_to_json
    get_metrics = common_module.get_metrics
    project_root = common_module.project_root
    quick_eval_solver = common_module.quick_eval_solver

from lisco.nn.nn_models import (
    FeedforwardNN,
    FeedforwardNNConfig)
from lisco.nn.solver import (
    Solver,
    SolverDecomposed,
    #TrainLogger,
    #NLPConfig,
)
from nlp_handler import NLPHandler,NLPConfig
from lisco.training.training_utils import TrainLogger
from lisco.nn.predictor import load_predictor
from lisco.training.train_solver import SolverTrainConfig, train_solver
from models.cstr_cascade import CSTRParams
from models.cstr_nmpc import NMPCParams

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def plot_history(train_logger: TrainLogger, save_dir: Path, title: str = "Solver Training History") -> None:
    """Save separate solver training plots from TrainLogger history."""
    import matplotlib.pyplot as plt

    plot_specs = [
        ("loss", "Solver Training Loss", save_dir / "loss.png"),
        ("n_resample", "Number of Resamples", save_dir / "n_resample.png"),
        ("n_resample_tol", "Resamples by Tolerance", save_dir / "n_resample_tol.png"),
        ("n_resample_iter", "Resamples by Iteration Limit", save_dir / "n_resample_iter.png"),
    ]

    for key, fig_title, save_path in plot_specs:
        fig, _ = train_logger.plot_history(
            keys_to_plot=[key],
            save_path=save_path,
            x_key="epoch",
            title=fig_title,
        )
        plt.close(fig)

    print(f"Training plots saved to: {save_dir}")

"""
def quick_eval(
    solver: torch.nn.Module,
    nlp,
    train_config: SolverTrainConfig,
    save_dir: Path,
    max_iter: int = 200,
) -> None:
   
    n_eval = int(train_config.batch_size)
    if getattr(nlp, "batch_size", None) != n_eval:
        raise RuntimeError(
            f"Expected nlp.batch_size={n_eval} for no-recompile quick eval, "
            f"got {getattr(nlp, 'batch_size', None)}."
        )

    def _plot_traj(traj: torch.Tensor, title: str, ylabel: str, save_path: Path) -> dict:
        if traj.ndim != 2:
            raise ValueError(f"Expected trajectory shape (T, B), got {tuple(traj.shape)}")

        def _kth_percentile(x: torch.Tensor, q: int, dim: int) -> torch.Tensor:
            n = int(x.shape[dim])
            if n < 1:
                raise ValueError("Cannot compute percentiles on empty dimension.")
            k = max(1, min(n, int(q / 100.0 * n)))
            return x.kthvalue(k, dim=dim).values

        metrics = {
            "min": traj.min(dim=1).values,
            "median": traj.median(dim=1).values,
            "max": traj.max(dim=1).values,
        }
        for q in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]:
            key = "median" if q == 50 else f"p{q}"
            metrics[key] = _kth_percentile(traj, q=q, dim=1)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        iters = list(range(traj.shape[0]))
        grey_percentiles = [5, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 95]
        for q in grey_percentiles:
            ax.plot(iters, metrics[f"p{q}"].detach().cpu(), color="grey", alpha=0.25, linewidth=1.0)
        ax.plot(iters, metrics["min"].detach().cpu(), label="min")
        ax.plot(iters, metrics["p1"].detach().cpu(), label="p1")
        ax.plot(iters, metrics["p10"].detach().cpu(), label="p10")
        ax.plot(iters, metrics["median"].detach().cpu(), label="median")
        ax.plot(iters, metrics["p90"].detach().cpu(), label="p90")
        ax.plot(iters, metrics["p99"].detach().cpu(), label="p99")
        ax.plot(iters, metrics["max"].detach().cpu(), label="max")
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return metrics

    with torch.no_grad():
        def _print_final_percentiles(label: str, metrics: dict) -> None:
            print(f"{label} trajectory stats (final iteration):")
            ordered_keys = [
                "min",
                "p1",
                "p5",
                "p10",
                "p15",
                "p20",
                "p25",
                "p30",
                "p35",
                "p40",
                "p45",
                "median",
                "p55",
                "p60",
                "p65",
                "p70",
                "p75",
                "p80",
                "p85",
                "p90",
                "p95",
                "p99",
                "max",
            ]
            for key in ordered_keys:
                print(f"  - {key:>6}: {metrics[key][-1].item():.3e}")

        p_eval = nlp.batch_gen_p(n_eval,p=0.0)
        z_solved, z_traj, step_times = solver.solve_batch(
            p_batch=p_eval,
            max_iter=max_iter,
            alpha=1.0,
            predictor=solver.predictor,
            update_mode="full",
            return_trajectory=True,
        )

        Tk_eval = nlp.Tk_batch_func(z_solved, p_eval)
        KKT_eval = nlp.KKT_batch_func(z_solved, p_eval)
        KKT_inf_norm = KKT_eval.norm(p=float("inf"), dim=1)
        tk_metrics = get_metrics(Tk_eval)
        kkt_final_metrics = get_metrics(KKT_inf_norm)

        # KKT inf-norm trajectory statistics across the evaluation batch.
        kkt_inf_traj = []
        g_pos_inf_traj = []
        h_inf_traj = []
        lam_neg_inf_traj = []
        comp_slack_inf_traj = []
        grad_L_inf_traj = []

        for z_k in z_traj:
            # KKT residual (evaluation metric)
            kkt_k = nlp.KKT_batch_func(z_k, p_eval)
            kkt_inf_traj.append(kkt_k.norm(p=float("inf"), dim=1))

            # Feasibility violations
            w_k, _nu_k, lam_k = nlp.extract_primal_dual_batch(z_k)

            if getattr(nlp, "n_g", 0) > 0:
                g_k = nlp.g_batch_func(w_k, p_eval)
                g_pos_k = torch.clamp(g_k, min=0.0)
                g_pos_inf_traj.append(g_pos_k.norm(p=float("inf"), dim=1))
            else:
                g_pos_inf_traj.append(torch.zeros(z_k.shape[0], device=z_k.device, dtype=z_k.dtype))

            if getattr(nlp, "n_h", 0) > 0:
                h_k = nlp.h_batch_func(w_k, p_eval)
                h_inf_traj.append(h_k.norm(p=float("inf"), dim=1))
            else:
                h_inf_traj.append(torch.zeros(z_k.shape[0], device=z_k.device, dtype=z_k.dtype))

            lam_neg_k = torch.clamp(-lam_k, min=0.0)
            lam_neg_inf_traj.append(lam_neg_k.norm(p=float("inf"), dim=1))

            comp_slack_k = nlp.comp_slack_batch_func(z_k, p_eval)
            comp_slack_inf_traj.append(comp_slack_k.norm(p=float("inf"), dim=1))

            grad_L_k = nlp.grad_L_batch_func(z_k, p_eval)
            grad_L_inf_traj.append(grad_L_k.norm(p=float("inf"), dim=1))

        kkt_inf_traj = torch.stack(kkt_inf_traj, dim=0)  # (T, B)
        g_pos_inf_traj = torch.stack(g_pos_inf_traj, dim=0)  # (T, B)
        h_inf_traj = torch.stack(h_inf_traj, dim=0)  # (T, B)
        lam_neg_inf_traj = torch.stack(lam_neg_inf_traj, dim=0)  # (T, B)
        comp_slack_inf_traj = torch.stack(comp_slack_inf_traj, dim=0)  # (T, B)
        grad_L_inf_traj = torch.stack(grad_L_inf_traj, dim=0)  # (T, B)

        kkt_traj_metrics = _plot_traj(
            kkt_inf_traj,
            title="Quick Eval KKT inf-norm Trajectory",
            ylabel="KKT inf-norm",
            save_path=save_dir / "quick_eval_kkt_inf_trajectory.png",
        )
        g_pos_traj_metrics = _plot_traj(
            g_pos_inf_traj,
            title="Quick Eval ||max(g, 0)||_inf Trajectory",
            ylabel="||max(g, 0)||_inf",
            save_path=save_dir / "quick_eval_g_pos_inf_trajectory.png",
        )
        h_traj_metrics = _plot_traj(
            h_inf_traj,
            title="Quick Eval ||h||_inf Trajectory",
            ylabel="||h||_inf",
            save_path=save_dir / "quick_eval_h_inf_trajectory.png",
        )
        lam_neg_traj_metrics = _plot_traj(
            lam_neg_inf_traj,
            title="Quick Eval ||min(lam, 0)||_inf Trajectory",
            ylabel="||min(lam, 0)||_inf",
            save_path=save_dir / "quick_eval_lam_neg_inf_trajectory.png",
        )
        comp_slack_traj_metrics = _plot_traj(
            comp_slack_inf_traj,
            title="Quick Eval ||lam * g||_inf Trajectory",
            ylabel="||lam * g||_inf",
            save_path=save_dir / "quick_eval_comp_slack_inf_trajectory.png",
        )
        grad_L_traj_metrics = _plot_traj(
            grad_L_inf_traj,
            title="Quick Eval ||grad L||_inf Trajectory",
            ylabel="||grad L||_inf",
            save_path=save_dir / "quick_eval_grad_L_inf_trajectory.png",
        )

        print(f"Quick eval samples (train batch size): {n_eval}")
        print(f"Solver iterations: {max_iter} (update_mode='line_search')")
        print("Merit function Tk (after solve):")
        print(f"  - Mean:   {tk_metrics['mean']:.3e}")
        print(f"  - Std:    {tk_metrics['std']:.3e}")
        print(f"  - Min:    {tk_metrics['min']:.3e}")
        print(f"  - Median: {tk_metrics['median']:.3e}")
        print(f"  - Max:    {tk_metrics['max']:.3e}")
        print("KKT inf-norm (after solve):")
        print(f"  - Mean:   {kkt_final_metrics['mean']:.3e}")
        print(f"  - Std:    {kkt_final_metrics['std']:.3e}")
        print(f"  - Min:    {kkt_final_metrics['min']:.3e}")
        print(f"  - Median: {kkt_final_metrics['median']:.3e}")
        print(f"  - Max:    {kkt_final_metrics['max']:.3e}")
        _print_final_percentiles("KKT inf-norm", kkt_traj_metrics)
        _print_final_percentiles("||max(g,0)||_inf", g_pos_traj_metrics)
        _print_final_percentiles("||h||_inf", h_traj_metrics)
        _print_final_percentiles("||min(lam,0)||_inf", lam_neg_traj_metrics)
        _print_final_percentiles("||lam * g||_inf", comp_slack_traj_metrics)
        _print_final_percentiles("||grad L||_inf", grad_L_traj_metrics)
        if step_times:
            print(f"  - Avg step time: {sum(step_times)/len(step_times):.3e} s")
        print(f"KKT trajectory figure saved to: {save_dir / 'quick_eval_kkt_inf_trajectory.png'}")
        print(f"g-pos trajectory figure saved to: {save_dir / 'quick_eval_g_pos_inf_trajectory.png'}")
        print(f"h trajectory figure saved to: {save_dir / 'quick_eval_h_inf_trajectory.png'}")
        print(f"lam-neg trajectory figure saved to: {save_dir / 'quick_eval_lam_neg_inf_trajectory.png'}")
        print(f"complementarity trajectory figure saved to: {save_dir / 'quick_eval_comp_slack_inf_trajectory.png'}")
        print(f"stationarity trajectory figure saved to: {save_dir / 'quick_eval_grad_L_inf_trajectory.png'}")

"""
def quick_eval(
    solver: torch.nn.Module,
    nlp,
    train_config: SolverTrainConfig,
    save_dir: Path,
    max_iter: int = 200,
) -> None:
    """Evaluate relaxed roots and random children warm-started from their parents."""
    n_eval = int(train_config.batch_size)

    if getattr(nlp, "batch_size", None) != n_eval:
        raise RuntimeError(
            f"Expected nlp.batch_size={n_eval}, "
            f"got {getattr(nlp, 'batch_size', None)}."
        )

    def _plot_trajectory(
        trajectory: torch.Tensor,
        title: str,
        ylabel: str,
        save_path: Path,
    ) -> dict[str, torch.Tensor]:
        if trajectory.ndim != 2:
            raise ValueError(
                f"Expected trajectory with shape (iterations, samples), "
                f"got {tuple(trajectory.shape)}"
            )

        if trajectory.shape[1] == 0:
            raise ValueError("Cannot plot a trajectory without valid samples.")

        quantiles = [
            0.01,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.99,
        ]

        metrics = {
            "min": trajectory.min(dim=1).values,
            "max": trajectory.max(dim=1).values,
        }

        for quantile in quantiles:
            percentile = int(round(100 * quantile))
            key = "median" if percentile == 50 else f"p{percentile}"
            metrics[key] = torch.quantile(
                trajectory,
                q=quantile,
                dim=1,
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        iterations =list(range(trajectory.shape[0]))

        grey_percentiles = [
            5,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            95,
        ]

        for percentile in grey_percentiles:
            ax.plot(
                iterations,
                metrics[f"p{percentile}"].detach().cpu(),
                color="grey",
                alpha=0.25,
                linewidth=1.0,
            )

        ax.plot(iterations, metrics["min"].detach().cpu(), label="min")
        ax.plot(iterations, metrics["p1"].detach().cpu(), label="p1")
        ax.plot(iterations, metrics["p10"].detach().cpu(), label="p10")
        ax.plot(iterations, metrics["median"].detach().cpu(), label="median")
        ax.plot(iterations, metrics["p90"].detach().cpu(), label="p90")
        ax.plot(iterations, metrics["p99"].detach().cpu(), label="p99")
        ax.plot(iterations, metrics["max"].detach().cpu(), label="max")

        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

        return metrics

    def _print_final_percentiles(
        label: str,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        ordered_keys = [
            "min",
            "p1",
            "p5",
            "p10",
            "p15",
            "p20",
            "p25",
            "p30",
            "p35",
            "p40",
            "p45",
            "median",
            "p55",
            "p60",
            "p65",
            "p70",
            "p75",
            "p80",
            "p85",
            "p90",
            "p95",
            "p99",
            "max",
        ]

        print(f"{label} at the final solver iteration:")

        for key in ordered_keys:
            print(f"  - {key:>6}: {metrics[key][-1].item():.3e}")

    def _evaluate_case(
        *,
        case_name: str,
        p_eval: torch.Tensor,
        valid_mask: torch.Tensor,
        predictor=None,
        z0_batch: torch.Tensor | None = None,
    ) -> dict:
        """Solve and evaluate one full-batch test case.

        NLP functions are always called on the complete configured batch.
        valid_mask is applied only after those evaluations.
        """
        z_solved, z_trajectory, step_times = solver.solve_batch(
            p_batch=p_eval,
            z0_batch=z0_batch,
            predictor=predictor,
            max_iter=max_iter,
            alpha=1.0,
            update_mode="full",
            return_trajectory=True,
        )

        Tk_full = nlp.Tk_batch_func(z_solved, p_eval)
        KKT_full = nlp.KKT_batch_func(z_solved, p_eval)
        KKT_inf_full = KKT_full.norm(p=float("inf"), dim=1)

        n_valid = int(valid_mask.sum().item())

        result = {
            "z_solved": z_solved,
            "z_trajectory": z_trajectory,
            "step_times": step_times,
            "Tk_full": Tk_full,
            "KKT_inf_full": KKT_inf_full,
            "valid_mask": valid_mask,
        }

        print("\n" + "=" * 70)
        print(f"Quick evaluation: {case_name}")
        print("=" * 70)
        print(f"Valid samples: {n_valid}/{n_eval}")
        print(f"Solver iterations: {max_iter}")
        print("Update mode: full")

        if n_valid == 0:
            print("No valid samples available; skipping statistics and plots.")
            return result

        Tk_valid = Tk_full[valid_mask]
        KKT_inf_valid = KKT_inf_full[valid_mask]

        tk_metrics = get_metrics(Tk_valid)
        kkt_final_metrics = get_metrics(KKT_inf_valid)

        kkt_inf_trajectory = []
        g_pos_inf_trajectory = []
        h_inf_trajectory = []
        lam_neg_inf_trajectory = []
        comp_slack_inf_trajectory = []
        grad_L_inf_trajectory = []

        for z_k in z_trajectory:
            # Evaluate every NLP function on the full compiled batch first.
            KKT_k_full = nlp.KKT_batch_func(z_k, p_eval)
            KKT_inf_k_full = KKT_k_full.norm(
                p=float("inf"),
                dim=1,
            )
            kkt_inf_trajectory.append(
                KKT_inf_k_full[valid_mask]
            )

            w_k, _nu_k, lam_k = nlp.extract_primal_dual_batch(z_k)

            if getattr(nlp, "n_g", 0) > 0:
                g_k_full = nlp.g_batch_func(w_k, p_eval)
                g_pos_inf_full = torch.clamp(
                    g_k_full,
                    min=0.0,
                ).norm(
                    p=float("inf"),
                    dim=1,
                )
            else:
                g_pos_inf_full = torch.zeros(
                    z_k.shape[0],
                    device=z_k.device,
                    dtype=z_k.dtype,
                )

            g_pos_inf_trajectory.append(
                g_pos_inf_full[valid_mask]
            )

            if getattr(nlp, "n_h", 0) > 0:
                h_k_full = nlp.h_batch_func(w_k, p_eval)
                h_inf_full = h_k_full.norm(
                    p=float("inf"),
                    dim=1,
                )
            else:
                h_inf_full = torch.zeros(
                    z_k.shape[0],
                    device=z_k.device,
                    dtype=z_k.dtype,
                )

            h_inf_trajectory.append(
                h_inf_full[valid_mask]
            )

            lam_neg_inf_full = torch.clamp(
                -lam_k,
                min=0.0,
            ).norm(
                p=float("inf"),
                dim=1,
            )
            lam_neg_inf_trajectory.append(
                lam_neg_inf_full[valid_mask]
            )

            comp_slack_k_full = nlp.comp_slack_batch_func(
                z_k,
                p_eval,
            )
            comp_slack_inf_full = comp_slack_k_full.norm(
                p=float("inf"),
                dim=1,
            )
            comp_slack_inf_trajectory.append(
                comp_slack_inf_full[valid_mask]
            )

            grad_L_k_full = nlp.grad_L_batch_func(
                z_k,
                p_eval,
            )
            grad_L_inf_full = grad_L_k_full.norm(
                p=float("inf"),
                dim=1,
            )
            grad_L_inf_trajectory.append(
                grad_L_inf_full[valid_mask]
            )

        trajectories = {
            "kkt_inf": torch.stack(
                kkt_inf_trajectory,
                dim=0,
            ),
            "g_pos_inf": torch.stack(
                g_pos_inf_trajectory,
                dim=0,
            ),
            "h_inf": torch.stack(
                h_inf_trajectory,
                dim=0,
            ),
            "lam_neg_inf": torch.stack(
                lam_neg_inf_trajectory,
                dim=0,
            ),
            "comp_slack_inf": torch.stack(
                comp_slack_inf_trajectory,
                dim=0,
            ),
            "grad_L_inf": torch.stack(
                grad_L_inf_trajectory,
                dim=0,
            ),
        }

        plot_definitions = {
            "kkt_inf": (
                "KKT inf-norm",
                "KKT inf-norm",
            ),
            "g_pos_inf": (
                "||max(g, 0)||_inf",
                "||max(g, 0)||_inf",
            ),
            "h_inf": (
                "Equality residual inf-norm",
                "||h||_inf",
            ),
            "lam_neg_inf": (
                "Negative multiplier violation",
                "||min(lambda, 0)||_inf",
            ),
            "comp_slack_inf": (
                "Complementarity residual",
                "||lambda * g||_inf",
            ),
            "grad_L_inf": (
                "Stationarity residual",
                "||grad L||_inf",
            ),
        }

        trajectory_metrics = {}

        for key, trajectory in trajectories.items():
            plot_title, ylabel = plot_definitions[key]
            save_path = (
                save_dir
                / f"quick_eval_{case_name}_{key}_trajectory.png"
            )

            trajectory_metrics[key] = _plot_trajectory(
                trajectory,
                title=f"{case_name}: {plot_title}",
                ylabel=ylabel,
                save_path=save_path,
            )

            print(f"Saved: {save_path}")

        print("Merit function Tk after solve:")
        print(f"  - Mean:   {tk_metrics['mean']:.3e}")
        print(f"  - Std:    {tk_metrics['std']:.3e}")
        print(f"  - Min:    {tk_metrics['min']:.3e}")
        print(f"  - Median: {tk_metrics['median']:.3e}")
        print(f"  - Max:    {tk_metrics['max']:.3e}")

        print("KKT inf-norm after solve:")
        print(f"  - Mean:   {kkt_final_metrics['mean']:.3e}")
        print(f"  - Std:    {kkt_final_metrics['std']:.3e}")
        print(f"  - Min:    {kkt_final_metrics['min']:.3e}")
        print(f"  - Median: {kkt_final_metrics['median']:.3e}")
        print(f"  - Max:    {kkt_final_metrics['max']:.3e}")

        n_converged = int(
            (Tk_valid < train_config.Tk_lim).sum().item()
        )
        print(
            f"Tk < {train_config.Tk_lim:.1e}: "
            f"{n_converged}/{n_valid}"
        )

        _print_final_percentiles(
            f"{case_name} KKT inf-norm",
            trajectory_metrics["kkt_inf"],
        )

        if step_times:
            print(
                "Average solver-step time: "
                f"{sum(step_times) / len(step_times):.3e} s"
            )

        result["trajectories"] = trajectories
        result["trajectory_metrics"] = trajectory_metrics
        return result

    with torch.no_grad():
        # --------------------------------------------------------------
        # Case 1: completely relaxed roots initialized by the predictor.
        # --------------------------------------------------------------
        p_root = nlp.batch_gen_p(
            n_eval,
            p=0.0,
        )

        all_roots_valid = torch.ones(
            n_eval,
            device=p_root.device,
            dtype=torch.bool,
        )

        root_result = _evaluate_case(
            case_name="relaxed",
            p_eval=p_root,
            valid_mask=all_roots_valid,
            predictor=solver.predictor,
            z0_batch=None,
        )

        z_root = root_result["z_solved"]
        Tk_root = root_result["Tk_full"]

        root_converged = Tk_root < train_config.Tk_lim

        # --------------------------------------------------------------
        # Case 2: one random bound fixing, initialized from the solved
        # parent primal-dual iterate rather than from the predictor.
        # --------------------------------------------------------------
        p_child, has_free,branch_indices,branch_values = nlp.branch_random_child_batch(
            p_root
        )
        #selected_branch_indices = branch_indices[has_free]
        #selected_branch_values = branch_values[has_free].to(
        #    dtype=z_root.dtype
        #)

        #binary_primal_offset = (
        #        nlp.N * nlp.nx
        #        + nlp.N * nlp.nmpc.nu
        #)

        #z_root[
        #    :,
        #    binary_primal_offset + selected_branch_indices,
        #] = selected_branch_values
        # Only children of genuinely converged roots count toward child
        # evaluation statistics. The solver still evaluates a full batch.
        valid_child = root_converged & has_free

        child_result = _evaluate_case(
            case_name="child",
            p_eval=p_child,
            valid_mask=valid_child,
            predictor=None,
            z0_batch=z_root,
        )

        Tk_child = child_result["Tk_full"]

        n_root_converged = int(root_converged.sum().item())
        n_valid_child = int(valid_child.sum().item())
        n_child_converged = int(
            (
                (Tk_child < train_config.Tk_lim)
                & valid_child
            ).sum().item()
        )

        print("\n" + "=" * 70)
        print("Combined quick-evaluation summary")
        print("=" * 70)
        print(
            f"Relaxed roots converged: "
            f"{n_root_converged}/{n_eval}"
        )
        print(
            f"Valid parent-warm-started children: "
            f"{n_valid_child}/{n_eval}"
        )
        print(
            f"Children converged: "
            f"{n_child_converged}/{n_valid_child}"
            if n_valid_child > 0
            else "Children converged: no valid children"
        )
def _build_solver(
    architecture: str,
    nn_model: torch.nn.Module,
    nlp,
    predictor,
    solver_cfg: dict,
    decomp_cfg: dict | None,
):
    gamma = solver_cfg.get("gamma", 0.1)
    convexification = solver_cfg.get("convexification", False)
    jvp_linesearch = solver_cfg.get("jvp_linesearch", False)

    if architecture == "monolithic":
        solver = Solver(
            model=nn_model,
            nlp=nlp,
            predictor=predictor,
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )
    elif architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        solver = SolverDecomposed(
            model=nn_model,
            nlp=nlp,
            n_nodes=decomp_cfg["n_nodes"],
            n_slide_steps=decomp_cfg["n_slide_steps"],
            overlap_merge_mode=decomp_cfg.get("overlap_merge_mode", "average"),
            predictor=predictor,
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected 'monolithic' or 'decomposed'.")

    if "norm_Fk_max" in solver_cfg:
        solver.norm_Fk_max = solver_cfg["norm_Fk_max"]

    return solver


def _solver_nn_dims(architecture: str, nlp, decomp_cfg: dict | None) -> tuple[int, int]:
    if architecture == "monolithic":
        return nlp.n_p + nlp.n_z + 1, nlp.n_z

    if architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        n_nodes = int(decomp_cfg["n_nodes"])
        n_in = nlp.n_p + n_nodes * nlp.n_z_stage + n_nodes + 2 * n_nodes
        n_out = n_nodes * nlp.n_z_stage
        return n_in, n_out

    raise ValueError(f"Unknown architecture '{architecture}'.")


def _load_predictor_init(predictor_path: str | None, nlp):
    if predictor_path is None:
        return None

    predictor_dir = Path(predictor_path)
    if not predictor_dir.is_absolute():
        predictor_dir = (project_root() / predictor_dir).resolve()

    predictor = load_predictor(predictor_dir,nlp)

    if int(predictor.nlp.n_p) != int(nlp.n_p):
        raise ValueError(
            f"Predictor n_p mismatch: predictor has {predictor.nlp.n_p}, solver has {nlp.n_p}."
        )
    if int(predictor.nlp.n_z) != int(nlp.n_z):
        raise ValueError(
            f"Predictor n_z mismatch: predictor has {predictor.nlp.n_z}, solver has {nlp.n_z}."
        )
    if int(predictor.nlp.N) != int(nlp.N):
        raise ValueError(
            f"Predictor horizon mismatch: predictor has N={predictor.nlp.N}, solver has N={nlp.N}."
        )
    if str(predictor.nlp.device) != str(nlp.device):
        raise ValueError(
            f"Predictor device mismatch: predictor has device='{predictor.nlp.device}', "
            f"solver has device='{nlp.device}'."
        )

    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad = False

    return predictor


def run_solver_training(config: dict) -> tuple[torch.nn.Module, TrainLogger, object, Path]:
    """Train a CSTR solver (monolithic or decomposed) from a dict config.

    Optional:
        predictor_path: path to a saved predictor run directory for initialization.
    """
    #model_params = config["model"]
    #nmpc_params = config["nmpc"]
    nlp_config = config["nlp"]
    nn_config = config["nn"]
    train_config = config["train"]
    architecture = config.get("architecture", "monolithic")
    solver_cfg = dict(config.get("solver", {}))
    decomp_cfg = config.get("decomposition")
    predictor_path = config.get("predictor_path")
    save_dir_str = config.get("save_dir", "results/solver_training")

    print("=" * 60)
    print("CSTR Cascade Solver Training")
    print(f"  architecture={architecture}")
    print(f"  predictor_init={'yes' if predictor_path else 'no'}")
    if architecture == "decomposed" and decomp_cfg is not None:
        print(
            f"  decomposition=(n_nodes={decomp_cfg['n_nodes']}, "
            f"n_slide_steps={decomp_cfg['n_slide_steps']}, "
            f"merge={decomp_cfg.get('overlap_merge_mode', 'average')})"
        )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup CSTR model and NMPC
    # -------------------------------------------------------------------------
    print("\n[1/4] Setting up CSTR model and NMPC...")
    nmpc = Four_Tank()#Linearized_Four_Tank()#Four_Tank()#Linearized_Four_Tank()#Nonlinear_double()#QP_Problem()#Linearized_Four_Tank()#Optimization_Problem()
    nlp = NLPHandler(
        nmpc,
        device=nlp_config.device,
        dtype=nlp_config.dtype,
        eps=nlp_config.eps,
        sigma=nlp_config.sigma,
        n_threads=nlp_config.n_threads,
        distributed_KKT_functions=nlp_config.distributed_KKT_functions,
        regularize=nlp_config.regularize,
        rho=nlp_config.rho
    )
    #nmpc, nlp = setup_nlp_from_params(model_params, nmpc_params, nlp_config)

    #print(f"  - Reactors: {model_params.nr}")
    #print(f"  - Horizon: {nmpc_params.N}")
    #print(f"  - Weights: w_R={nmpc_params.w_R}, w_S={nmpc_params.w_S}")
    #print(f"  - Objective: {nmpc_params.objective}")
    print(f"  - n_p (parameters): {nlp.n_p}")
    print(f"  - n_z (primal-dual): {nlp.n_z}")
    if architecture == "decomposed":
        print(f"  - n_z_stage: {nlp.n_z_stage}")
    print(f"  - Device: {nlp.device}")

    # -------------------------------------------------------------------------
    # 2. Create neural network and solver
    # -------------------------------------------------------------------------
    print("\n[2/4] Creating neural network and solver...")
    predictor = _load_predictor_init(predictor_path, nlp)
    if predictor is not None:
        print(f"  - Warm-start predictor loaded from: {predictor_path}")

    n_in, n_out = _solver_nn_dims(architecture, nlp, decomp_cfg)

    nn_model = FeedforwardNN(
        n_in=n_in,
        n_out=n_out,
        **dataclasses.asdict(nn_config),
    ).to(nlp_config.device)

    n_params = sum(p.numel() for p in nn_model.parameters())
    print(
        f"  - Architecture: {n_in} -> {nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {n_out}"
    )
    print(f"  - Parameters: {n_params:,}")
    print(f"  - Activation: {nn_config.act_fn}")

    solver = _build_solver(
        architecture=architecture,
        nn_model=nn_model,
        nlp=nlp,
        predictor=predictor,
        solver_cfg=solver_cfg,
        decomp_cfg=decomp_cfg,
    )

    # -------------------------------------------------------------------------
    # 3. Train
    # -------------------------------------------------------------------------
    print("\n[3/4] Training solver...")
    # solver.model.parametrize_output()

    train_logger = train_solver(nlp,solver, config=train_config)
    #load_path_dir = Path("results/solver_training/20260820_153119_mono_pred")
    #nlp.batch_size=256
    #nlp.setup_for_batch(256)
    #if not load_path_dir.is_absolute():
    #    load_dir = (project_root() / load_path_dir).resolve()
    #solver.load(load_dir,nlp)
    # solver.model.remove_parametrizations()

    # -------------------------------------------------------------------------
    # 4. Save results + quick evaluation
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if architecture == "decomposed" and decomp_cfg is not None:
        suffix = (
            f"_decomp_n{decomp_cfg['n_nodes']}"
            f"_s{decomp_cfg['n_slide_steps']}"
            f"_{decomp_cfg.get('overlap_merge_mode', 'average')}"
        )
    else:
        suffix = "_mono"
    if predictor is not None:
        suffix += "_pred"

    save_dir = Path(save_dir_str)
    if not save_dir.is_absolute():
        save_dir = project_root() / save_dir
    save_dir = save_dir / f"{timestamp}{suffix}"#nr{model_params.nr}_N{nmpc_params.N}
    save_dir.mkdir(parents=True, exist_ok=True)

    solver.save(save_dir)
    train_logger.save_history(save_dir, file_name="training_log")
    with open(save_dir / "config.json", "w") as f:
        json.dump(dict_config_to_json(config), f, indent=2)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    quick_eval(solver, nlp, train_config=train_config, save_dir=save_dir, max_iter=200)
    print(f"\nResults saved to: {save_dir}")

    return solver, train_logger, nlp, save_dir


if __name__ == "__main__":
    config = {
        "architecture": "monolithic", # "monolithic", "decomposed"
        "model": CSTRParams(nr=2, dt=1.0),
        "nmpc": NMPCParams(N=100, w_R=10.0, w_S=1.0, w_uA=0.1, w_uB=0.0),
        "nlp": NLPConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float64,
            rho=1e-2,
            regularize=True,
            sigma=1e-2,
            eps=1e-8,
            distributed_KKT_functions=False,
        ),
        "nn": FeedforwardNNConfig(
            n_neurons=4096,
            n_hidden_layers=1,
            act_fn="gelu",
            nn_dtype=torch.float32,
        ),
        "solver": {
            "gamma": 0.1,
            "convexification": False,
            "norm_Fk_max": 1.0,
            "jvp_linesearch": False,
        },
        # Optional predictor run directory for initialization
        "predictor_path": "results/predictor_training/20260918_090053__single",#20260912_125717__single",#20260903_234244__single",#"results/predictor_training/20260907_151401__single",#20260906_210734__single",#20260903_110924__single",#20260827_101507__single",#20260827_101507__single",#20260826_192453__single",#20260825_210057__single",#20260825_163325__single",#20260824_222730__single",#20260824_203650__single",#20260824_134536__single",#20260821_081231__single",#20260821_135515__single",#20260821_081231__single"#20260819_145734__single",#20260818_174858__single",#20260818_111422__single",#20260817_151525__single",#20260805_084456__single",#20260626_220030__single",#20260606_095445__single",
        # Only used when architecture == "decomposed"
        "decomposition": {
            "n_nodes": 5,
            "n_slide_steps": 1,
            "overlap_merge_mode": "average",
        },
        "train": SolverTrainConfig(
            N_epochs=100000,
            batch_size= 256,
            lr=5*1e-5,
            weight_decay=0.0,
            use_amsgrad=True,
            resampling_strategy="all",#"tree_trajectory", #"optimal_buffer", "all"
            Tk_lim=1e-8,
            n_steps_max_total=1000,
            n_steps_max_no_improvement=10,
            # buffer_max_size=10000,
            # p_buffer=1.0,
            # p_fresh_min=0.1,
            use_adaptive_clip_grad=True,
            print_frequency=10,
            log_values=True,
        ),
        "save_dir": "results/solver_training",
    }

    solver, train_logger, nlp, save_dir = run_solver_training(config)
    plot_history(train_logger, save_dir)
