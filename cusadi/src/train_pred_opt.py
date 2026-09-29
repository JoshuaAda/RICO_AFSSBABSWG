"""Predictor training script for CSTR Cascade Economic NMPC.

Supports:
- single_stage Predictor
- two_stage PredictorTwoStage
- decomposed PredictorDecomposed

Usage:
    python experiments/train_pred_cstr.py
"""

from __future__ import annotations
import numpy as np
import dataclasses
import json
from datetime import datetime
from pathlib import Path
from optimization import Optimization_Problem,Linearized_Four_Tank,QP_Problem,Nonlinear_double,Four_Tank
import torch
from copy import deepcopy
try:
    from .common import (
        setup_nlp_from_params,
        dict_config_to_json,
        project_root,
        plot_predictor_results,
    )
except ImportError:
    # Fallback: manually load common.py from the same directory
    import sys
    import importlib.util
    from pathlib import Path as PathlibPath
    
    common_path = PathlibPath(__file__).parent / "common.py"
    spec = importlib.util.spec_from_file_location("common", common_path)
    common_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common_module)
    
    setup_nlp_from_params = common_module.setup_nlp_from_params
    dict_config_to_json = common_module.dict_config_to_json
    project_root = common_module.project_root
    plot_predictor_results = common_module.plot_predictor_results
from lisco.nn.nn_models import (
    FeedforwardNN,
    FeedforwardNNConfig)
from nlp_handler import NLPHandler, NLPConfig
from lisco.nn.predictor import (
    Predictor,
    PredictorDecomposed,
    PredictorTwoStage,
    PredictorConfig)
from lisco.training.training_utils import (
    TrainLogger)
from nlp_handler import NLPConfig
from lisco.training.train_predictor import PredictorTrainConfig, train_predictor
from models.cstr_cascade import CSTRParams
from models.cstr_nmpc import NMPCParams

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def _predictor_nn_dims(
    architecture: str,
    nlp,
    decomp_cfg: dict | None,
) -> tuple[int, int]:
    if architecture == "single_stage":
        return nlp.n_p, nlp.n_z

    if architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        n_nodes = int(decomp_cfg["n_nodes"])
        n_in = nlp.n_p + 2 * n_nodes
        n_out = n_nodes * nlp.n_z_stage
        return n_in, n_out

    raise ValueError(f"Unknown architecture '{architecture}'.")


def _build_predictor(
    architecture: str,
    nn_model: torch.nn.Module,
    nlp,
    pred_cfg: PredictorConfig | None,
    decomp_cfg: dict | None,
):
    if architecture == "single_stage":
        if pred_cfg is None:
            pred_cfg = PredictorConfig()
        return Predictor(model=nn_model, nlp=nlp, **dataclasses.asdict(pred_cfg))

    if architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        return PredictorDecomposed(
            model=nn_model,
            nlp=nlp,
            n_nodes=decomp_cfg["n_nodes"],
            n_slide_steps=decomp_cfg["n_slide_steps"],
            overlap_merge_mode=decomp_cfg.get("overlap_merge_mode", "average"),
        )

    raise ValueError(f"Unknown architecture '{architecture}'.")


# =============================================================================
# Experiment Runner
# =============================================================================


def run_predictor_training(
    config: dict,
) -> tuple[torch.nn.Module, TrainLogger, object, Path]:
    """Train a CSTR predictor (single_stage, two_stage, or decomposed).

    Args:
        config: Dict with keys:
            "model"    - CSTRParams
            "nmpc"     - NMPCParams
            "nlp"      - NLPConfig
            "architecture" - "single_stage" | "two_stage" | "decomposed"
            "nn"       - FeedforwardNNConfig
            "pred"     - PredictorConfig (used for single_stage/two_stage)
            "decomposition" - dict for decomposed architecture
            "train"    - PredictorTrainConfig
            "save_dir" - str, base directory for results (default: "results/predictor_training")

    Returns:
        Tuple of (trained predictor, training logger, nlp handler, save directory path).
    """
    #model_params = config["model"]
    #nmpc_params  = config["nmpc"]
    nlp_config   = config["nlp"]
    architecture = config.get("architecture", "single_stage")
    if architecture not in ("single_stage", "two_stage", "decomposed"):
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Expected 'single_stage', 'two_stage', or 'decomposed'."
        )
    decomp_cfg   = config.get("decomposition")
    if architecture == "decomposed" and decomp_cfg is None:
        raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
    nn_config    = config["nn"]
    pred_config  = config.get("pred")
    if architecture in ("single_stage", "two_stage") and pred_config is None:
        pred_config = PredictorConfig()
    train_config = config["train"]
    save_dir_str = config.get("save_dir", "results/predictor_training")

    print("=" * 60)
    print("CSTR Cascade Predictor Training")
    print(f"  architecture={architecture}")
    if architecture == "decomposed" and decomp_cfg is not None:
        print(
            f"  decomposition=(n_nodes={decomp_cfg['n_nodes']}, "
            f"n_slide_steps={decomp_cfg['n_slide_steps']}, "
            f"merge={decomp_cfg.get('overlap_merge_mode', 'average')})"
        )
    elif pred_config is not None:
        print(
            f"  w_sigmoid={pred_config.w_sigmoid}, lam_softplus={pred_config.lam_softplus}, "
            f"use_gate={pred_config.use_gate}, tau={pred_config.tau}"
        )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup CSTR model and NMPC
    # -------------------------------------------------------------------------
    print("\n[1/4] Setting up CSTR model and NMPC...")
    nmpc=Four_Tank()#Linearized_Four_Tank()#Four_Tank()#Linearized_Four_Tank()#Nonlinear_double()#QP_Problem()#Optimization_Problem()
    nlp = NLPHandler(
        nmpc,
        device=nlp_config.device,
        dtype=nlp_config.dtype,
        eps=nlp_config.eps,
        sigma=nlp_config.sigma,
        n_threads=nlp_config.n_threads,
        distributed_KKT_functions=nlp_config.distributed_KKT_functions,
    )
    #nmpc, nlp = setup_nlp_from_params(model_params, nmpc_params, nlp_config)

    #print(f"  - Reactors: {model_params.nr}")
    #print(f"  - Horizon: {nmpc_params.N}")
    #print(f"  - Weights: w_R={nmpc_params.w_R}, w_S={nmpc_params.w_S}")
    #print(f"  - Objective: {nmpc_params.objective}")
    print(f"  - n_p (parameters): {nlp.n_p}")
    print(f"  - n_z (primal-dual): {nlp.n_z}")
    print(f"  - n_g (inequalities): {nlp.n_g}")
    if architecture == "decomposed":
        print(f"  - n_z_stage: {nlp.n_z_stage}")
    print(f"  - Device: {nlp.device}")

    # -------------------------------------------------------------------------
    # 2. Create neural network and Predictor
    # -------------------------------------------------------------------------
    print("\n[2/4] Creating neural network and Predictor...")
    if architecture == "two_stage":
        if pred_config is None:
            raise ValueError("Missing PredictorConfig for architecture='two_stage'.")
        nn_w_model = FeedforwardNN(
            n_in=nlp.n_p,
            n_out=nlp.n_w,
            **dataclasses.asdict(nn_config),
        ).to(nlp_config.device)

        nn_dual_model = FeedforwardNN(
            n_in=nlp.n_p + nlp.n_w,
            n_out=nlp.n_h + nlp.n_g,
            **dataclasses.asdict(nn_config),
        ).to(nlp_config.device)

        n_params_w = sum(p.numel() for p in nn_w_model.parameters())
        n_params_dual = sum(p.numel() for p in nn_dual_model.parameters())
        print(
            f"  - Shared NN config (both stages): "
            f"{nn_config.n_neurons}x{nn_config.n_hidden_layers}, act={nn_config.act_fn}"
        )
        print(
            f"  - Stage1 (p->w): {nlp.n_p} -> {nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {nlp.n_w}"
        )
        print(
            f"  - Stage2 ([p,w]->[nu,lam]): {nlp.n_p + nlp.n_w} -> "
            f"{nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {nlp.n_h + nlp.n_g}"
        )
        print(f"  - Parameters stage1: {n_params_w:,}")
        print(f"  - Parameters stage2: {n_params_dual:,}")
        print(f"  - Total parameters: {n_params_w + n_params_dual:,}")

        predictor = PredictorTwoStage(
            model_w=nn_w_model,
            model_dual=nn_dual_model,
            nlp=nlp,
            **dataclasses.asdict(pred_config),
        )
    else:
        n_in, n_out = _predictor_nn_dims(architecture, nlp, decomp_cfg)
        nn_model = FeedforwardNN(
            n_in=n_in,
            n_out=n_out,
            **dataclasses.asdict(nn_config),
        )
        nn_model = nn_model.to(nlp_config.device)

        n_params = sum(p.numel() for p in nn_model.parameters())
        print(
            f"  - Architecture: {n_in} -> {nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {n_out}"
        )
        print(f"  - Parameters: {n_params:,}")
        print(f"  - Activation: {nn_config.act_fn}")

        predictor = _build_predictor(
            architecture=architecture,
            nn_model=nn_model,
            nlp=nlp,
            pred_cfg=pred_config,
            decomp_cfg=decomp_cfg,
        )

    # -------------------------------------------------------------------------
    # 3. Train
    # -------------------------------------------------------------------------
    print("\n[3/4] Training Predictor...")
    train_logger = train_predictor(predictor, nlp, train_config)


    # -------------------------------------------------------------------------
    # 4. Save results
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if architecture == "decomposed" and decomp_cfg is not None:
        suffix = (
            f"_decomp_n{decomp_cfg['n_nodes']}"
            f"_s{decomp_cfg['n_slide_steps']}"
            f"_{decomp_cfg.get('overlap_merge_mode', 'average')}"
        )
    elif architecture == "two_stage":
        suffix = "_2stage"
    else:
        suffix = "_single"

    save_dir = Path(save_dir_str)
    if not save_dir.is_absolute():
        save_dir = project_root() / save_dir
    save_dir = save_dir / f"{timestamp}_{suffix}"#{timestamp}
    save_dir.mkdir(parents=True, exist_ok=True)

    predictor.save(save_dir)
    train_logger.save_history(save_dir, file_name="training_log")
    with open(save_dir / "config.json", "w") as f:
        json.dump(dict_config_to_json(config), f, indent=2)

    #predictor_dir = Path("results/predictor_training/__single")
    #if not predictor_dir.is_absolute():
    #    predictor_dir = (project_root() / predictor_dir).resolve()
    #predictor_new = _build_predictor(
    #    architecture=architecture,
    #   nn_model=deepcopy(nn_model),
    #    nlp=nlp,
    #    pred_cfg=pred_config,
    #    decomp_cfg=decomp_cfg,
    #)

    #predictor_new=predictor_new.load(predictor_dir, nlp)

    # Quick evaluation
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    with torch.no_grad():
        n_test = 100
        nlp.setup_for_batch(n_test)

        p_test = nlp.batch_gen_p(n_test)#torch.Tensor(np.repeat(np.array([-2,2,1,-1]),100).reshape((4,100)).T).to("cuda").double()#
        z_pred = predictor.predict(p_test).to(dtype=nlp_config.dtype)
        #z_pred_new=predictor_new.predict(p_test).to(dtype=nlp_config.dtype)
        Tk_test = nlp.Tk_batch_func(z_pred, p_test)
        KKT_test = nlp.KKT_batch_func(z_pred, p_test)
        KKT_inf_norm = KKT_test.norm(p=float("inf"), dim=1)

        print(f"Test samples: {n_test}")
        print(f"Merit function Tk:")
        print(f"  - Mean:   {Tk_test.mean().item():.3e}")
        print(f"  - Std:    {Tk_test.std().item():.3e}")
        print(f"  - Min:    {Tk_test.min().item():.3e}")
        print(f"  - Max:    {Tk_test.max().item():.3e}")
        print(f"KKT inf-norm:")
        print(f"  - Mean:   {KKT_inf_norm.mean().item():.3e}")
        print(f"  - Median: {KKT_inf_norm.median().item():.3e}")
        print(f"  - Max:    {KKT_inf_norm.max().item():.3e}")

        if (
            architecture in ("single_stage", "two_stage")
            and pred_config is not None
            and pred_config.lam_softplus
        ):
            _, _, lam_pred = nlp.extract_primal_dual_batch(z_pred)
            print(f"Lambda statistics:")
            print(f"  - Mean:   {lam_pred.mean().item():.3e}")
            print(f"  - Min:    {lam_pred.min().item():.3e}")
            print(f"  - Max:    {lam_pred.max().item():.3e}")
            print(f"  - Frac zero (<1e-8): {(lam_pred.abs() < 1e-8).float().mean().item():.3f}")

        print(f"\nResults saved to: {save_dir}")

    return predictor, train_logger, nlp, save_dir


# =============================================================================
# Main Script
# =============================================================================


if __name__ == "__main__":
    # Old single-stage config (kept for reference)
    # config = {
    #     "architecture": "single_stage",
    #     "model":  CSTRParams(nr=1, dt=1.0),
    #     "nmpc":   NMPCParams(N=5, w_R=10.0, w_S=1.0, w_uA=0.1, w_uB=0.0),
    #     "nlp":    NLPConfig(
    #         sigma=0.0,
    #         eps=1e-3,
    #     ),
    #     "nn":     FeedforwardNNConfig(n_neurons=512, n_hidden_layers=1, act_fn="gelu",
    #                                   nn_dtype=torch.float32),
    #     "pred":   PredictorConfig(w_sigmoid=True, lam_softplus=True),
    #     "train":  PredictorTrainConfig(N_epochs=5000, lr=1e-3, use_adaptive_clip_grad=True),
    #     "save_dir": "results/predictor_training",
    # }

    config = {
        "architecture": "single_stage",  # "single_stage", "two_stage", "decomposed"
        "model":  CSTRParams(nr=2, dt=1.0),
        "nmpc":   NMPCParams(N=100, w_R=10.0, w_S=1.0, w_uA=0.1, w_uB=0.0),
        "nlp":    NLPConfig(
            sigma=1e-2,
            eps=1e-8,
            rho=1e-2,
            regularize=True,
            distributed_KKT_functions=False,
        ),
        # Shared NN config (reused for both subnetworks in two-stage mode)
        "nn":   FeedforwardNNConfig(
            n_neurons=4096,
            n_hidden_layers=1,
            act_fn="gelu",
            nn_dtype=torch.float32,
        ),
        # Shared post-processing
        "pred":   PredictorConfig(
            w_sigmoid=False,
            lam_softplus=False,
            use_gate=False,
            tau=0.01,
        ),
        "train":  PredictorTrainConfig(
            N_epochs=30000,
            batch_size=256,
            lr=1e-3,
            use_adaptive_clip_grad=True,
            use_fixed_val_batch=False,
        ),
        "save_dir": "results/predictor_training",
        # Only used when architecture == "decomposed"
        "decomposition": {
            "n_nodes": 5,
            "n_slide_steps": 1,
            "overlap_merge_mode": "average",
        },
    }

    predictor, train_logger, nlp, save_dir = run_predictor_training(config)

    # Plot training loss
    architecture = config.get("architecture", "single_stage")
    if architecture == "decomposed":
        title = "PredictorDecomposed Training Loss"
    elif architecture == "two_stage":
        title = "PredictorTwoStage Training Loss"
    else:
        title = "Predictor Training Loss"
    plot_predictor_results(train_logger, save_dir, title=title)
