from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from casadi import DM

from branching.branching_dataset import BranchingExampleWriter
from main import Node, branch_and_bound_solver
from optimization import Four_Tank,Optimization_Problem,Linearized_Four_Tank,Nonlinear_double


def solve_dataset_node(solver, x0, node_id, max_iter=200):
    """Solve one relaxation and retain LISCO feasibility."""
    node = solver.node_list[node_id]
    parameter = np.concatenate((x0, node.lbz, node.ubz))
    parameter_batch = np.repeat(parameter, solver.current_batch_size)
    parameter_batch = parameter_batch.reshape(
        (-1, solver.current_batch_size)
    ).T
    parameter_batch = torch.as_tensor(
        parameter_batch,
        dtype=solver.lisco.nlp.dtype,
        device=solver.device,
    )

    z_solved, _ = solver.lisco.solve_batch(
        p_batch=parameter_batch,
        max_iter=max_iter,
        alpha=1.0,
        predictor=solver.predictor,
        update_mode="full",
        return_trajectory=False,
    )
    residual = solver.lisco.nlp.Tk_batch_func(
        z_solved,
        parameter_batch,
    ).reshape(-1)[0]
    feasible = bool(torch.isfinite(residual) and residual <= 1e-4)

    if not feasible:
        node.lb = np.inf
        node.pruned = True
        return False

    node.lb = float(
        solver.get_cost_from_z(
            z_solved,
            parameter_batch,
            solver.current_batch_size,
        )[0][0][0]
    )
    node.var = solver.get_primal_from_z(z_solved, 0)
    node.zk=z_solved[0]
    solver.check_current_sol(node_id)
    return True


def initialize_tree(solver, x0, node_iterations):
    solver.node_list = []
    solver.upper = np.inf
    solver.upper_var = 0
    solver.nodes_visited = 0

    solver.node_list.append(
        Node(
            np.zeros((solver.N * solver.nz, 1)),
            np.ones((solver.N * solver.nz, 1)),
        )
    )
    if not solve_dataset_node(
        solver,
        x0,
        0,
        max_iter=node_iterations,
    ):
        return False

    solver.generate_upper_bound(x0, lisco=True)
    return np.isfinite(float(solver.upper))


def add_child(
    solver,
    *,
    parent_id,
    branch_variable,
    branch_value,
    lower_bounds,
    upper_bounds,
    branched_list,
    x0,
    node_iterations,
):
    parent = solver.node_list[parent_id]
    solver.node_list.append(
        Node(
            lower_bounds,
            upper_bounds,
            depth=parent.depth + 1,
            parent_id=parent_id,
            branch_var=branch_variable,
            branch_value=branch_value,
            branched_list=branched_list,
        )
    )
    child_id = len(solver.node_list) - 1
    solve_dataset_node(
        solver,
        x0,
        child_id,
        max_iter=node_iterations,
    )
    return child_id


def collect_instance(
    solver,
    *,
    x0,
    instance_id,
    max_nodes,
    teacher_iterations,
    node_iterations,
    min_teacher_score_span,
):
    if not initialize_tree(solver, x0, node_iterations):
        return None

    examples = []
    nodes_visited = 0
    infeasible_nodes = 0
    skipped_labels = 0

    while nodes_visited < max_nodes:
        node_id = solver.decide_next_node()
        if node_id is None:
            break

        node = solver.node_list[node_id]
        if node.is_integer or node.pruned:
            continue

        nodes_visited += 1
        try:
            teacher = solver.evaluate_branching_teacher_parallel(
                node_id,
                x0,
                num_candidates=solver.num_candidates,
                max_iter=teacher_iterations,
            )
        except RuntimeError:
            node.pruned = True
            infeasible_nodes += 1
            continue

        candidate_indices = teacher["candidate_indices"]
        teacher_scores = np.asarray(
            teacher["teacher_scores"],
            dtype=np.float64,
        ).reshape(-1)
        finite_scores = teacher_scores[np.isfinite(teacher_scores)]
        informative_label = (
            finite_scores.shape[0] >= 2
            and np.ptp(finite_scores) > min_teacher_score_span
        )
        if informative_label:
            examples.append({
                "features": torch.as_tensor(
                    teacher["features"],
                    dtype=torch.float32,
                ),
                "teacher_scores": torch.as_tensor(
                    teacher_scores,
                    dtype=torch.float32,
                ),
                "candidate_indices": torch.as_tensor(
                    candidate_indices,
                    dtype=torch.long,
                ),
                "instance_id": instance_id,
                "x0": torch.as_tensor(
                    np.asarray(x0).reshape(-1),
                    dtype=torch.float32,
                ),
                "node_depth": int(node.depth),
                "feature_names": solver.get_branching_feature_names(),
            })
        else:
            skipped_labels += 1

        best_var_combo = teacher["best_var_combo"]
        branched_list = node.branched_list.copy()
        for variable_index in best_var_combo:
            branched_list.append(variable_index)

        for pattern_index, (lower_bounds, upper_bounds) in enumerate(
            teacher["best_combo_bounds"]
        ):
            solver.node_list.append(
                Node(
                    lower_bounds,
                    upper_bounds,
                    depth=node.depth + 1,
                    parent_id=node_id,
                    branch_var=best_var_combo,
                    branch_value=None,
                    branched_list=branched_list,
                )
            )
            child = solver.node_list[-1]
            child.var = teacher["best_combo_solutions"][pattern_index]
            child.lb = teacher["best_combo_costs"][pattern_index]
            child.zk = teacher["zk"][pattern_index]
            solver.check_current_sol(len(solver.node_list) - 1)
            infeasible_nodes += int(
                not teacher["best_combo_feasible"][pattern_index]
            )

        node.branched = True
        solver.prune_nodes()

    return {
        "examples": examples,
        "nodes_visited": nodes_visited,
        "infeasible_nodes": infeasible_nodes,
        "skipped_labels": skipped_labels,
    }


def state_constraint_bounds(problem):
    #if not problem.x_lb or not problem.x_ub:
    #    raise ValueError("Optimization problem does not expose state bounds")
    lower = np.array([[-1],[-1],[-1],[-1]])#np.asarray(DM(problem.x_lb[0]), dtype=np.float64).reshape(-1)
    upper = np.array([[1],[1],[1],[1]])#np.asarray(DM(problem.x_ub[0]), dtype=np.float64).reshape(-1)
    #if lower.shape != (problem.nx,) or upper.shape != (problem.nx,):
    #    raise ValueError("Initial-state bounds do not match problem.nx")
    #if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
    #    raise ValueError("Initial-state sampling requires finite state bounds")
    #if np.any(lower > upper):
    #    raise ValueError("Initial-state lower bounds exceed upper bounds")
    return lower, upper


def sample_x0(rng, lower, upper):
    return rng.uniform(lower, upper).reshape(-1, 1)


def generate(args):
    problem = Four_Tank()#Linearized_Four_Tank()#Nonlinear_double()#Linearized_Four_Tank()#Optimization_Problem()
    x0_lower, x0_upper = state_constraint_bounds(problem)

    solver = branch_and_bound_solver(
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
        num_candidates=args.num_candidates,
        device=args.device,
    )

    rng = np.random.default_rng(args.seed)
    writer = BranchingExampleWriter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    attempts = 0
    max_attempts = (
        args.max_sampling_attempts
        if args.max_sampling_attempts is not None
        else 10 * args.num_instances
    )

    while accepted < args.num_instances and attempts < max_attempts:
        attempts += 1
        x0 = sample_x0(rng, x0_lower, x0_upper)
        result = collect_instance(
            solver,
            x0=x0,
            instance_id=accepted,
            max_nodes=args.max_nodes_per_instance,
            teacher_iterations=args.teacher_iterations,
            node_iterations=args.node_iterations,
            min_teacher_score_span=args.min_teacher_score_span,
        )

        if result is None or not result["examples"]:
            reason = (
                "no feasible root/incumbent"
                if result is None
                else "no informative branching labels"
            )
            print(
                f"attempt={attempts} rejected={reason} "
                f"x0={x0.reshape(-1).tolist()}"
            )
            continue

        writer.examples.extend(result["examples"])
        writer.save(args.output)
        accepted += 1
        print(
            f"instance={accepted}/{args.num_instances} attempt={attempts} "
            f"examples={len(result['examples'])} "
            f"nodes_visited={result['nodes_visited']} "
            f"infeasible_nodes={result['infeasible_nodes']} "
            f"skipped_labels={result['skipped_labels']} "
            f"total_examples={len(writer.examples)} "
            f"x0={x0.reshape(-1).tolist()}"
        )

    if accepted < args.num_instances:
        raise RuntimeError(
            f"Generated only {accepted}/{args.num_instances} feasible, "
            f"informative instances after {attempts} attempts"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate learned-branching labels with parallel LISCO strong "
            "branching and initial states sampled from the model constraints."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--max-nodes-per-instance", type=int, default=100)
    parser.add_argument("--teacher-iterations", type=int, default=200)
    parser.add_argument("--node-iterations", type=int, default=200)
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--min-teacher-score-span", type=float, default=1e-6)
    parser.add_argument("--max-sampling-attempts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
