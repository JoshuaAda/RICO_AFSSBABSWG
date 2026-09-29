"""Closed-loop case study for the linearized four-tank MIQP.

The experiment runs many independent receding-horizon trajectories.  At each
closed-loop step it solves the current MPC problem, applies only the first
input, simulates the plant once, and records the realized quantities.  The
stored stage cost is therefore computed from the realized ``(x_k, u_k, z_k)``
and is not the open-loop objective returned by the optimizer.

Examples
--------
Run the Gurobi reference implementation::

    python case_study.py --backend gurobi

Run Bonmin::

    python case_study.py --backend bonmin

Run the LISCO branch-and-bound implementation configured in ``main.py``::

    python case_study.py --backend lisco --lisco-max-iter 500

Results are saved as a pickle (complete nested dictionary), a compressed NPZ
(numeric trajectory arrays), and a small JSON summary.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from optimization import Four_Tank,Linearized_Four_Tank


Backend = Literal["lisco", "bonmin", "gurobi"]


# Keep the realized plant and stage cost explicit.  These are the matrices used
# by Linearized_Four_Tank in optimization.py.
A_FOUR_TANK = np.array(
    [
        [0.923, 0.0, 0.181, 0.0],
        [0.0, 0.946, 0.0, 0.149],
        [0.0, 0.0, 0.811, 0.0],
        [0.0, 0.0, 0.0, 0.846],
    ],
    dtype=np.float64,
)
B_FOUR_TANK = np.array(
    [
        [0.283, 0.023],
        [0.012, 0.306],
        [0.0, 0.216],
        [0.144, 0.0],
    ],
    dtype=np.float64,
)
Q_FOUR_TANK = np.diag([1.0, 1.0, 0.0, 0.0])
R_FOUR_TANK = 0.1 * np.eye(2)
C_Z_FOUR_TANK = np.array([0.1, 0.2], dtype=np.float64)


@dataclass(frozen=True)
class CaseStudyConfig:
    backend: Backend = "gurobi"
    n_trajectories: int = 100
    trajectory_length: int = 10
    seed: int = 0
    initial_state_low: float = -1.0
    initial_state_high: float = 1.0
    output_dir: Path = Path("results/case_study")
    lisco_max_iter: int = 500
    lisco_num_candidates: int = 40
    device: str = "cuda"
    gurobi_output: bool = False


def simulate_four_tank(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply the exact linear dynamics used by Linearized_Four_Tank."""
    h_bar = np.array([12.4, 12.7, 1.8, 1.4])
    phi = np.array([[
        2.0 * np.sqrt(h_bar[0])
        * (np.sqrt(h_bar[0] + x[0]) - np.sqrt(h_bar[0]))],
        [
        2.0 * np.sqrt(h_bar[1])
        * (np.sqrt(h_bar[1] + x[1]) - np.sqrt(h_bar[1]))],
        [
        2.0 * np.sqrt(h_bar[2])
        * (np.sqrt(h_bar[2] + x[2]) - np.sqrt(h_bar[2]))],
        [
        2.0 * np.sqrt(h_bar[3])
        * (np.sqrt(h_bar[3] + x[3]) - np.sqrt(h_bar[3]))]]
    )
    return np.array([
        x[0]
        + (A_FOUR_TANK[0, 0] - 1.0) * phi[0]
        + A_FOUR_TANK[0, 2] * phi[2],

        x[1]
        + (A_FOUR_TANK[1, 1] - 1.0) * phi[1]
        + A_FOUR_TANK[1, 3] * phi[3],

        x[2]
        + (A_FOUR_TANK[2, 2] - 1.0) * phi[2],

        x[3]
        + (A_FOUR_TANK[3, 3] - 1.0) * phi[3]]
    ).squeeze() + B_FOUR_TANK @ u
    #return A_FOUR_TANK @ x + B_FOUR_TANK @ u


def realized_stage_cost(x: np.ndarray, u: np.ndarray, z: np.ndarray) -> float:
    """Evaluate x'Qx + u'Ru + c'z for the realized closed-loop values."""
    return float(x @ Q_FOUR_TANK @ x + u @ R_FOUR_TANK @ u + C_Z_FOUR_TANK @ z)


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_node_count(stats: dict[str, Any]) -> float:
    """Best-effort extraction because CasADi plugins use different names."""
    candidate_keys = (
        "node_count",
        "nodes",
        "mip_node_count",
        "bb_nodes",
        "n_nodes",
        "iter_count",
    )
    for key in candidate_keys:
        if key in stats:
            return _to_float(stats[key])
    for container_key in ("iterations", "solver_stats"):
        nested = stats.get(container_key)
        if isinstance(nested, dict):
            value = _extract_node_count(nested)
            if np.isfinite(value):
                return value
    return np.nan


class GurobiBackend:
    """Persistent Gurobi MIQP whose initial-state bounds change each step."""

    name = "gurobi"

    def __init__(self, problem: Four_Tank, output: bool = False):
        if problem.model_gurobi is None or problem.model_gurobi_vars is None:
            raise RuntimeError("Gurobi model is unavailable. Check gurobipy and its license.")
        self.model = problem.model_gurobi
        self.variables = problem.model_gurobi_vars
        self.model.Params.OutputFlag = int(output)

    def solve(self, x: np.ndarray) -> dict[str, Any]:
        X = self.variables["X"]
        for index, value in enumerate(x):
            X[0, index].LB = float(value)
            X[0, index].UB = float(value)

        start = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - start
        #print(elapsed)
        # Gurobi status 2 is OPTIMAL.  Accepting a merely feasible incumbent
        # would make comparisons with Bonmin/LISCO ambiguous.
        if int(self.model.Status) != 2:
            return {
                "success": False,
                "status": str(self.model.Status),
                "message": "Gurobi did not return an optimal solution",
                "solve_time": elapsed,
                "nodes": _to_float(getattr(self.model, "NodeCount", np.nan)),
                "objective": np.nan,
            }

        return {
            "success": True,
            "status": "optimal",
            "message": "",
            "solve_time": elapsed,
            "nodes": _to_float(self.model.NodeCount),
            "objective": _to_float(self.model.ObjVal),
            "u0": np.asarray(self.variables["U"].X[0], dtype=np.float64),
            "z0": np.asarray(self.variables["Z"].X[0], dtype=np.float64),
            "z_pred": np.asarray(self.variables["Z"].X[0:40], dtype=np.float64)
        }


class BonminBackend:
    """CasADi/Bonmin baseline using the problem definition from optimization.py."""

    name = "bonmin"

    def __init__(self, problem: Four_Tank):
        from casadi import nlpsol

        self.problem = problem
        self.solver = nlpsol(
            "case_study_bonmin",
            "bonmin",
            problem.prob,
            {"discrete": problem.discrete},
        )

    def solve(self, x: np.ndarray) -> dict[str, Any]:
        problem = self.problem
        z_lb = np.zeros((problem.N * problem.nz, 1))
        z_ub = np.ones((problem.N * problem.nz, 1))
        parameters = np.concatenate((x.reshape(-1, 1), z_lb, z_ub), axis=0)

        start = time.perf_counter()
        try:

            import io
            import re
            from contextlib import redirect_stdout, redirect_stderr

            log = io.StringIO()

            with redirect_stdout(log), redirect_stderr(log):
                result = self.solver(p=parameters, lbg=problem.lbg, ubg=problem.ubg)
                elapsed = time.perf_counter() - start
                # print(elapsed)
                stats = self.solver.stats()

            bonmin_log = log.getvalue()
            print(bonmin_log)
            patterns = [
                r"and (\d+) nodes",
            ]

            n_nodes = np.nan
            for pattern in patterns:
                match = re.search(pattern, bonmin_log, flags=re.IGNORECASE)
                if match:
                    n_nodes = int(match.group(1))
                    break
        except Exception as exc:  # Preserve partial case-study data on failures.
            return {
                "success": False,
                "status": "exception",
                "message": repr(exc),
                "solve_time": time.perf_counter() - start,
                "nodes": np.nan,
                "objective": np.nan,
            }

        success = bool(stats.get("success", False))
        solution = np.asarray(result["x"], dtype=np.float64).reshape(-1)
        u_offset = problem.N * problem.nx
        z_offset = u_offset + problem.N * problem.nu
        return {
            "success": success,
            "status": str(stats.get("return_status", "unknown")),
            "message": "" if success else str(stats),
            "solve_time": elapsed,
            "nodes": n_nodes,#_extract_node_count(stats),
            "objective": _to_float(result["f"]),
            "u0": solution[u_offset : u_offset + problem.nu],
            "z0": solution[z_offset : z_offset + problem.nz],
            "z_pred": solution[z_offset: z_offset + problem.nz * problem.N]
        }


class LiscoBackend:
    """Adapter around the learned branch-and-bound solver configured in main.py."""

    name = "lisco"

    def __init__(
        self,
        problem: Four_Tank,
        max_iter: int,
        num_candidates: int,
        device: str,
    ):
        # Import lazily: main.py imports optional solver dependencies and loads
        # the predictor/LISCO checkpoints in branch_and_bound_solver.__init__.
        from main import branch_and_bound_solver

        self.problem = problem
        self.solver = branch_and_bound_solver(
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            problem.lbz,
            problem.ubz,
            problem.prob,
            problem.discrete,
            problem.model_gurobi,
            problem.model_gurobi_vars,
            problem.nx,
            problem.nu,
            problem.nz,
            problem.N,
            r=1,
            num_candidates=num_candidates,
            device=device,
            max_iter=max_iter,
        )

    def solve(self, x: np.ndarray) -> dict[str, Any]:
        # CUDA calls are asynchronous.  Synchronizing around the solve makes the
        # wall-clock measurement comparable to CPU-based solvers.
        try:
            import torch

            if torch.cuda.is_available() and str(self.solver.device).startswith("cuda"):
                torch.cuda.synchronize()
        except ImportError:
            torch = None
        #self.solver.load_branching_policy("branching/branching/results/branching_policy_new_nonlinear_tank_100.pt")#")#new_40.pt")#nonlinear_tank_100.pt")#40.pt")
        start = time.perf_counter()
        try:
            primal, objective, nodes, reported_time,Tk_val =self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)##self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1),r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1), r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#self.solver.solve_lisco_parallel(x.reshape(-1, 1), r=1)#self.solver.solve_branch_and_bound_nn(x.reshape(-1, 1))#solve_lisco_parallel(x.reshape(-1, 1), r=1)#solve_branch_and_bound_nn(x.reshape(-1, 1))#solve_lisco_parallel(x.reshape(-1, 1), r=1)

            if torch is not None and torch.cuda.is_available() and str(self.solver.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        except Exception as exc:
            return {
                "success": False,
                "status": "exception",
                "message": repr(exc),
                "solve_time": reported_time,#time.perf_counter() - start,
                "nodes": np.nan,
                "objective": np.nan,
            }

        if primal is None or np.isscalar(primal):
            return {
                "success": False,
                "status": "no_incumbent",
                "message": "LISCO returned no primal incumbent",
                "solve_time": reported_time,#elapsed,
                "reported_solve_time": _to_float(reported_time),
                "nodes": _to_float(nodes),
                "Tk_val":Tk_val,
                "objective": _to_float(objective),
            }

        solution = np.asarray(primal, dtype=np.float64).reshape(-1)
        problem = self.problem
        u_offset = problem.N * problem.nx
        z_offset = u_offset + problem.N * problem.nu
        if solution.size < z_offset + problem.nz:
            return {
                "success": False,
                "status": "invalid_solution",
                "message": f"Expected at least {z_offset + problem.nz} primal entries; got {solution.size}",
                "solve_time": reported_time,#elapsed,
                "reported_solve_time": _to_float(reported_time),
                "nodes": _to_float(nodes),
                "objective": _to_float(objective),
            }

        return {
            "success": True,
            "status": "incumbent",
            "message": "",
            "solve_time": reported_time,#elapsed,
            "reported_solve_time": _to_float(reported_time),
            "nodes": _to_float(nodes),
            "objective": _to_float(objective),
            "Tk_val": Tk_val,
            "u0": solution[u_offset : u_offset + problem.nu],
            "z0": solution[z_offset : z_offset + problem.nz],
            "z_pred":solution[z_offset : z_offset + problem.nz*problem.N]
        }


def _make_backend(config: CaseStudyConfig, problem: Four_Tank):#Linearized_Four_Tank):
    if config.backend == "gurobi":
        return GurobiBackend(problem, output=config.gurobi_output)
    if config.backend == "bonmin":
        return BonminBackend(problem)
    if config.backend == "lisco":
        return LiscoBackend(
            problem,
            max_iter=config.lisco_max_iter,
            num_candidates=config.lisco_num_candidates,
            device=config.device,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def _empty_results(config: CaseStudyConfig, initial_states: np.ndarray) -> dict[str, Any]:
    n_traj = config.n_trajectories
    n_steps = config.trajectory_length
    return {
        "metadata": {
            "backend": config.backend,
            "n_trajectories": n_traj,
            "trajectory_length": n_steps,
            "seed": config.seed,
            "initial_state_low": config.initial_state_low,
            "initial_state_high": config.initial_state_high,
            "state_definition": "states[i, k] is the realized current state x_k",
            "stage_cost_definition": "x_k.T Q x_k + u_k.T R u_k + c_z.T z_k",
            "Q": Q_FOUR_TANK.copy(),
            "R": R_FOUR_TANK.copy(),
            "c_z": C_Z_FOUR_TANK.copy(),
            "A": A_FOUR_TANK.copy(),
            "B": B_FOUR_TANK.copy(),
        },
        "initial_states": initial_states.copy(),
        # Include the terminal realized state, hence n_steps + 1.
        "states": np.full((n_traj, n_steps + 1, 4), np.nan),
        "inputs": np.full((n_traj, n_steps, 2), np.nan),
        "binary_inputs": np.full((n_traj, n_steps, 2), np.nan),
        "z_pred":np.full((n_traj, n_steps, 20,2), np.nan),
        "stage_costs": np.full((n_traj, n_steps), np.nan),
        "solve_times": np.full((n_traj, n_steps), np.nan),
        "reported_solve_times": np.full((n_traj, n_steps), np.nan),
        "nodes_visited": np.full((n_traj, n_steps), np.nan),
        "open_loop_objectives": np.full((n_traj, n_steps), np.nan),
        "success": np.zeros((n_traj, n_steps), dtype=bool),
        "Tk_val":[],
        "status": [["not_run" for _ in range(n_steps)] for _ in range(n_traj)],
        "messages": [["" for _ in range(n_steps)] for _ in range(n_traj)],
    }


def run_case_study(config: CaseStudyConfig) -> dict[str, Any]:
    """Run and return the complete closed-loop experiment dictionary."""
    if config.n_trajectories <= 0 or config.trajectory_length <= 0:
        raise ValueError("n_trajectories and trajectory_length must be positive")
    if config.initial_state_low >= config.initial_state_high:
        raise ValueError("initial_state_low must be smaller than initial_state_high")

    rng = np.random.default_rng(config.seed)
    initial_states = rng.uniform(
        config.initial_state_low,
        config.initial_state_high,
        size=(config.n_trajectories, 4),
    )
    results = _empty_results(config, initial_states)

    problem = Four_Tank()#Linearized_Four_Tank()#Four_Tank()#Linearized_Four_Tank()
    backend = _make_backend(config, problem)

    for trajectory_index in range(config.n_trajectories):
        x = initial_states[trajectory_index].copy()#29].copy()#39
        results["states"][trajectory_index, 0] = x

        for step_index in range(config.trajectory_length):

            solved = backend.solve(x)
            results["solve_times"][trajectory_index, step_index] = solved["solve_time"]
            results["reported_solve_times"][trajectory_index, step_index] = solved.get(
                "reported_solve_time", np.nan
            )
            #results["Tk_val"].append(np.array(solved.get("Tk_val").cpu().detach()).squeeze())
            results["nodes_visited"][trajectory_index, step_index] = solved.get("nodes", np.nan)
            results["open_loop_objectives"][trajectory_index, step_index] = solved.get(
                "objective", np.nan
            )
            results["success"][trajectory_index, step_index] = bool(solved["success"])
            results["status"][trajectory_index][step_index] = str(solved.get("status", "unknown"))
            results["messages"][trajectory_index][step_index] = str(solved.get("message", ""))

            if not solved["success"]:
                # The remaining state/input sequence is undefined after a failed
                # MPC step, so retain NaNs and continue with the next trajectory.
                break

            u0 = np.asarray(solved["u0"], dtype=np.float64).reshape(2)
            z0 = np.asarray(solved["z0"], dtype=np.float64).reshape(2)
            z_pred=np.asarray(solved["z_pred"], dtype=np.float64).reshape(-1,2)
            results["inputs"][trajectory_index, step_index] = u0
            results["binary_inputs"][trajectory_index, step_index] = z0
            results["z_pred"][trajectory_index, step_index]=z_pred
            results["stage_costs"][trajectory_index, step_index] = realized_stage_cost(x, u0, z0)

            x = simulate_four_tank(x, u0)
            results["states"][trajectory_index, step_index + 1] = x

        completed = int(results["success"][trajectory_index].sum())
        print(
            f"[{trajectory_index + 1:3d}/{config.n_trajectories}] "
            f"completed_steps={completed}/{config.trajectory_length}"
        )

    valid_costs = results["stage_costs"][np.isfinite(results["stage_costs"])]
    valid_times = results["solve_times"][np.isfinite(results["solve_times"])]
    valid_nodes = results["nodes_visited"][np.isfinite(results["nodes_visited"])]
    trajectory_complete = results["success"].all(axis=1)
    results["summary"] = {
        "successful_steps": int(results["success"].sum()),
        "attempted_steps": int(np.count_nonzero(np.asarray(results["status"]) != "not_run")),
        "complete_trajectories": int(trajectory_complete.sum()),
        "failed_trajectories": int((~trajectory_complete).sum()),
        "mean_stage_cost": float(valid_costs.mean()) if valid_costs.size else np.nan,
        "mean_solve_time": float(valid_times.mean()) if valid_times.size else np.nan,
        "median_solve_time": float(np.median(valid_times)) if valid_times.size else np.nan,
        "max_solve_time": float(valid_times.max()) if valid_times.size else np.nan,
        "mean_nodes_visited": float(valid_nodes.mean()) if valid_nodes.size else np.nan,
        "total_realized_cost_per_trajectory": np.nansum(results["stage_costs"], axis=1),
    }
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = results["metadata"]["backend"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"linearized_four_tank_{backend}_{stamp}"

    pickle_path = output_dir / f"{stem}.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)

    numeric_keys = (
        "initial_states",
        "states",
        "inputs",
        "binary_inputs",
        "z_pred",
        "stage_costs",
        "solve_times",
        "reported_solve_times",
        "nodes_visited",
        "Tk_val",
        "open_loop_objectives",
        "success",
    )
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **{key: results[key] for key in numeric_keys})

    summary_path = output_dir / f"{stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe({"metadata": results["metadata"], "summary": results["summary"]}),
            stream,
            indent=2,
        )

    return {"pickle": pickle_path, "npz": npz_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("lisco", "bonmin", "gurobi"), default="lisco")
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-state-low", type=float, default=-1.0)
    parser.add_argument("--initial-state-high", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/case_study"))
    parser.add_argument("--lisco-max-iter", type=int, default=40)
    parser.add_argument("--lisco-num-candidates", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gurobi-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CaseStudyConfig(
        backend=args.backend,
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.seed,
        initial_state_low=args.initial_state_low,
        initial_state_high=args.initial_state_high,
        output_dir=args.output_dir,
        lisco_max_iter=args.lisco_max_iter,
        lisco_num_candidates=args.lisco_num_candidates,
        device=args.device,
        gurobi_output=args.gurobi_output,
    )
    results = run_case_study(config)
    paths = save_results(results, config.output_dir)

    print("\nCase-study summary")
    print(json.dumps(_json_safe(results["summary"]), indent=2))
    for kind, path in paths.items():
        print(f"{kind}: {path.resolve()}")


if __name__ == "__main__":
    main()
