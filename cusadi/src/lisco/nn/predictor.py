import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..nn.nn_models import FeedforwardNN
from optimization import Optimization_Problem

@dataclass
class PredictorConfig:
    """Post-processing config for Predictor output. Defaults match Predictor.__init__.

    Controls the output transformations applied after the raw NN forward pass.
    Note: model and nlp are excluded — they are always objects, not config values.
    convexification is excluded — it is not yet implemented.
    """

    w_sigmoid: bool = True
    lam_softplus: bool = True
    use_gate: bool = False
    tau: float = 0.01


class Predictor(torch.nn.Module):
    """Predictor class for Neural Network model that predicts primal-dual solutions.

    Args:
        model: Neural network that maps p -> z_raw = [w, nu, beta].
        nlp: NLPHandler instance (needed for z decomposition, Tk evaluation, and gate).
        w_sigmoid: Apply sigmoid to w output, scaling it to [0, 1].
        lam_softplus: Apply softplus to lambda output, ensuring lambda >= 0.
        use_gate: Apply constraint-activity gate a_i = sigmoid(g_i(w) / tau) to lambda.
        tau: Gate transition width. Only used if use_gate=True.
        convexification: Whether to use convexified Tk loss (not yet implemented).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        nlp,
        w_sigmoid: bool = True,
        lam_softplus: bool = True,
        use_gate: bool = False,
        tau: float = 0.01,
        convexification: bool = False,
    ):
        super().__init__()
        self.model = model
        self.nlp = nlp
        self.w_sigmoid = w_sigmoid
        self.lam_softplus = lam_softplus
        self.use_gate = use_gate
        self.tau = tau
        self.convexification = convexification
        self.offset = 1e-16

        if convexification:
            raise NotImplementedError("Convexified Tk not yet implemented in NLPHandler")

        print("----------------------------------")
        print(self)
        print("----------------------------------")

    @property
    def Tk_batch_func(self):
        """Always returns the current nlp.Tk_batch_func (cusadi GPU version after setup_for_batch)."""
        return self.nlp.Tk_batch_func

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        z_raw = self.model(p)

        if not (self.w_sigmoid or self.lam_softplus or self.use_gate):
            return z_raw

        w_raw, nu, beta = self.nlp.extract_primal_dual_batch(z_raw)

        w = torch.sigmoid(w_raw) if self.w_sigmoid else w_raw
        lam = F.softplus(beta) if self.lam_softplus else beta

        if self.use_gate:
            with torch.no_grad():
                g = self.nlp.g_batch_func(w, p)
                assert g.max() <= 1e-12, (
                    f"g must be non-positive (inequality feasibility), got max(g) = {g.max().item():.3e}"
                )
            a = torch.sigmoid(g / self.tau)
            lam = a * lam

        return torch.cat([w, nu, lam], dim=1)

    @torch.no_grad()
    def predict(self, p_batch: torch.Tensor) -> torch.Tensor:
        return self(p_batch)

    def save(self, save_dir):
        """Save predictor config (JSON) and model weights (.pt) to save_dir.

        Writes:
            predictor_config.json  — all hyperparameters, human-readable
            weights.pt             — FeedforwardNN state dict

        The full Predictor can be reconstructed with Predictor.load(save_dir).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            #"model_params": self.nlp.nmpc.model_params.to_dict(),
            #"nmpc_params":  self.nlp.nmpc.nmpc_params.to_dict(),
            "nlp_params": {
                "device":     self.nlp.device,
                "dtype":      str(self.nlp.dtype).split(".")[-1],
                "eps":        self.nlp.eps,
                "sigma":      self.nlp.sigma,
                "n_threads":  self.nlp.n_threads,
            },
            "nn_params": self.model.to_dict(),
            "predictor_params": {
                "w_sigmoid":      self.w_sigmoid,
                "lam_softplus":   self.lam_softplus,
                "use_gate":       self.use_gate,
                "tau":            self.tau,
                "convexification": self.convexification,
            },
        }

        with open(save_dir / "predictor_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(save_dir)

    @classmethod
    def load(cls, load_dir,nlp) -> "Predictor":
        """Reconstruct a Predictor from a directory written by save().

        Rebuilds the full chain: CSTRParams → NMPCParams → CSTRNMPC →
        NLPHandler → FeedforwardNN (with loaded weights) → Predictor.

        Note: hardcoded to CSTRNMPC; generalise if other NMPC types are added.
        """
        #from nlp_handler import NLPHandler
        #from src.models.cstr_cascade import CSTRParams
        #from src.models.cstr_nmpc import CSTRNMPC, NMPCParams

        load_dir = Path(load_dir)

        with open(load_dir / "predictor_config.json") as f:
            config = json.load(f)

        #model_params = CSTRParams.from_dict(config["model_params"])
        #nmpc_params  = NMPCParams.from_dict(config["nmpc_params"])
        #nmpc = CSTRNMPC(model_params, nmpc_params)

        nlp_cfg = dict(config["nlp_params"])
        nlp_cfg["dtype"] = getattr(torch, nlp_cfg["dtype"])
        nlp_cfg.pop("batch_size", None)   # compat: old files may contain this key
        print("\n[1/4] Setting up CSTR model and NMPC...")
        #nmpc = Optimization_Problem()
        #nlp = NLPHandler(
        #    nmpc,
        #    device=nlp_cfg["device"],
        #    dtype=nlp_cfg["dtype"],
        #    eps=nlp_cfg["eps"],
        #    sigma=nlp_cfg["sigma"],
        #    n_threads=nlp_cfg["n_threads"],
        #    distributed_KKT_functions=False#nlp_cfg["distributed_KKT_functions"],
        #)
        #nlp = NLPHandler(nmpc, **nlp_cfg)

        nn_cfg = dict(config["nn_params"])
        nn_cfg["nn_dtype"] = getattr(torch, nn_cfg["nn_dtype"])
        model = FeedforwardNN(**nn_cfg)
        model.load_weights(load_dir)

        model.to(nlp.device)

        return cls(model, nlp, **config["predictor_params"])


class PredictorDecomposed(torch.nn.Module):
    """Windowed decomposed predictor with overlap merging.

    The shared NN is evaluated in parallel over sliding windows of `n_nodes`
    stages. Consecutive windows advance by `n_slide_steps`.

    NN input per window:
        [p, positional encoding per covered stage (flattened)]

    NN output per window:
        [z_stage_0, ..., z_stage_{n_nodes-1}] (flattened)

    Overlapping stage predictions are merged using `overlap_merge_mode`.
    Currently, only overlap_merge_mode='average' is implemented.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        nlp,
        n_nodes: int,
        n_slide_steps: int,
        overlap_merge_mode: str = "average",
        convexification: bool = False,
    ):
        super().__init__()
        self.model = model
        self.nlp = nlp
        self.n_nodes = int(n_nodes)
        self.n_slide_steps = int(n_slide_steps)
        self.overlap_merge_mode = str(overlap_merge_mode)
        self.convexification = convexification
        self.offset = 1e-16

        if convexification:
            raise NotImplementedError("Convexified Tk not yet implemented in NLPHandler")

        self._validate_config(self.nlp.N)
        self.n_windows = 1 + (self.nlp.N - self.n_nodes) // self.n_slide_steps
        starts = (
            torch.arange(self.n_windows, device=self.nlp.device, dtype=torch.long)
            * self.n_slide_steps
        )
        offsets = torch.arange(self.n_nodes, device=self.nlp.device, dtype=torch.long)
        self._window_stage_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        self._window_stage_indices_flat = self._window_stage_indices.reshape(-1)
        self._avg_merge_counts = torch.bincount(
            self._window_stage_indices_flat,
            minlength=self.nlp.N,
        )

        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def _validate_config(self, N: int) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")
        if self.n_slide_steps < 1:
            raise ValueError(f"n_slide_steps must be >= 1, got {self.n_slide_steps}")
        if self.n_slide_steps > self.n_nodes:
            raise ValueError(
                f"n_slide_steps must be <= n_nodes to avoid gaps, got "
                f"n_slide_steps={self.n_slide_steps}, n_nodes={self.n_nodes}"
            )
        if N < self.n_nodes:
            raise ValueError(
                f"Invalid decomposition for horizon N={N}: n_nodes={self.n_nodes} exceeds N."
            )
        if (N - self.n_nodes) % self.n_slide_steps != 0:
            raise ValueError(
                f"Invalid decomposition for horizon N={N}: "
                f"(N - n_nodes) % n_slide_steps must be 0, got "
                f"({N} - {self.n_nodes}) % {self.n_slide_steps} = "
                f"{(N - self.n_nodes) % self.n_slide_steps}."
            )
        if self.overlap_merge_mode not in ("average", "handoff"):
            raise ValueError(
                f"Invalid overlap_merge_mode '{self.overlap_merge_mode}'. "
                "Expected one of: 'average', 'handoff'."
            )
        if self.overlap_merge_mode != "average":
            raise NotImplementedError(
                f"overlap_merge_mode='{self.overlap_merge_mode}' is not implemented. "
                "Use 'average'."
            )

    @property
    def Tk_batch_func(self):
        """Always returns the current nlp.Tk_batch_func (cusadi GPU version after setup_for_batch)."""
        return self.nlp.Tk_batch_func

    def _map_window_inputs(self, p: torch.Tensor) -> torch.Tensor:
        B = p.shape[0]
        N = self.nlp.N

        if N == 1:
            pos_all = torch.tensor([[0.0, 1.0]], device=p.device, dtype=p.dtype)
        else:
            k = torch.arange(N, device=p.device, dtype=p.dtype).view(N, 1)
            pos_forward = k / (N - 1)
            pos_backward = 1 - pos_forward
            pos_all = torch.cat([pos_forward, pos_backward], dim=1)

        pos_windows = pos_all[self._window_stage_indices]
        pos_windows = pos_windows.unsqueeze(0).expand(B, -1, -1, -1)
        pos_windows_flat = pos_windows.reshape(B, self.n_windows, 2 * self.n_nodes)

        p_expanded = p.unsqueeze(1).expand(B, self.n_windows, -1)
        nn_inputs = torch.cat([p_expanded, pos_windows_flat], dim=-1)
        return nn_inputs.reshape(B * self.n_windows, -1)

    def _merge_window_outputs_average(self, z_windows: torch.Tensor) -> torch.Tensor:
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage
        B = z_windows.shape[0]
        z_flat = z_windows.reshape(B, -1, n_z_stage)
        z_sum = torch.zeros(B, N, n_z_stage, device=z_windows.device, dtype=z_windows.dtype)

        stage_idx = self._window_stage_indices_flat.to(device=z_windows.device)
        stage_idx = stage_idx.view(1, -1, 1).expand(B, -1, n_z_stage)
        z_sum.scatter_add_(dim=1, index=stage_idx, src=z_flat)

        count = self._avg_merge_counts.to(device=z_windows.device, dtype=z_windows.dtype)
        return z_sum / count.view(1, N, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        B = p.shape[0]
        n_z_stage = self.nlp.n_z_stage

        nn_inputs_flat = self._map_window_inputs(p)
        z_windows_flat = self.model(nn_inputs_flat)
        z_windows = z_windows_flat.reshape(B, self.n_windows, self.n_nodes, n_z_stage)

        if self.overlap_merge_mode == "average":
            z_stages = self._merge_window_outputs_average(z_windows=z_windows)
        else:
            raise NotImplementedError

        return self.nlp.compose_z_batch(z_stages)

    @torch.no_grad()
    def predict(self, p_batch: torch.Tensor) -> torch.Tensor:
        return self(p_batch)

    def save(self, save_dir):
        """Save PredictorDecomposed config and model weights to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            #"model_params": self.nlp.nmpc.model_params.to_dict(),
            #"nmpc_params": self.nlp.nmpc.nmpc_params.to_dict(),
            "nlp_params": {
                "device": self.nlp.device,
                "dtype": str(self.nlp.dtype).split(".")[-1],
                "eps": self.nlp.eps,
                "sigma": self.nlp.sigma,
                "n_threads": self.nlp.n_threads,
            },
            "nn_params": self.model.to_dict(),
            "predictor_params": {
                "n_nodes": self.n_nodes,
                "n_slide_steps": self.n_slide_steps,
                "overlap_merge_mode": self.overlap_merge_mode,
                "convexification": self.convexification,
            },
        }

        with open(save_dir / "predictor_decomposed_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(save_dir)

    @classmethod
    def load(cls, load_dir) -> "PredictorDecomposed":
        """Reconstruct a PredictorDecomposed from a directory written by save()."""
        from nlp_handler import NLPHandler
        #from src.models.cstr_cascade import CSTRParams
        #from src.models.cstr_nmpc import CSTRNMPC, NMPCParams

        load_dir = Path(load_dir)

        with open(load_dir / "predictor_decomposed_config.json") as f:
            config = json.load(f)

        #model_params = CSTRParams.from_dict(config["model_params"])
        #nmpc_params = NMPCParams.from_dict(config["nmpc_params"])
        #nmpc = CSTRNMPC(model_params, nmpc_params)

        nlp_cfg = dict(config["nlp_params"])
        nlp_cfg["dtype"] = getattr(torch, nlp_cfg["dtype"])
        nlp_cfg.pop("batch_size", None)
        #nlp = NLPHandler(nmpc, **nlp_cfg)
        nmpc = Optimization_Problem()
        nlp = NLPHandler(
            nmpc,
            device=nlp_cfg.device,
            dtype=nlp_cfg.dtype,
            eps=nlp_cfg.eps,
            sigma=nlp_cfg.sigma,
            n_threads=nlp_cfg.n_threads,
            distributed_KKT_functions=nlp_cfg.distributed_KKT_functions,
        )
        nn_cfg = dict(config["nn_params"])
        nn_cfg["nn_dtype"] = getattr(torch, nn_cfg["nn_dtype"])
        model = FeedforwardNN(**nn_cfg)
        model.load_weights(load_dir)
        model.to(nlp.device)

        return cls(model, nlp, **config["predictor_params"])


class PredictorTwoStage(torch.nn.Module):
    """Two-stage predictor with separate primal and dual subnetworks.

    Stage 1: p -> w
    Stage 2: [p, w] -> [nu, beta], where beta is optionally mapped to lam via softplus.
    """

    def __init__(
        self,
        model_w: torch.nn.Module,
        model_dual: torch.nn.Module,
        nlp,
        w_sigmoid: bool = True,
        lam_softplus: bool = True,
        use_gate: bool = False,
        tau: float = 0.01,
        convexification: bool = False,
    ):
        super().__init__()
        self.model_w = model_w
        self.model_dual = model_dual
        self.nlp = nlp
        self.w_sigmoid = w_sigmoid
        self.lam_softplus = lam_softplus
        self.use_gate = use_gate
        self.tau = tau
        self.convexification = convexification
        self.offset = 1e-16

        if convexification:
            raise NotImplementedError("Convexified Tk not yet implemented in NLPHandler")

        print("----------------------------------")
        print(self)
        print("----------------------------------")

    @property
    def Tk_batch_func(self):
        """Always returns the current nlp.Tk_batch_func (cusadi GPU version after setup_for_batch)."""
        return self.nlp.Tk_batch_func

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        w_raw = self.model_w(p)
        w = torch.sigmoid(w_raw) if self.w_sigmoid else w_raw

        dual_input = torch.cat([p, w], dim=1)
        dual_raw = self.model_dual(dual_input)
        nu = dual_raw[:, : self.nlp.n_h]
        beta = dual_raw[:, self.nlp.n_h :]
        lam = F.softplus(beta) if self.lam_softplus else beta

        if self.use_gate:
            with torch.no_grad():
                g = self.nlp.g_batch_func(w, p)
                assert g.max() <= 1e-12, (
                    f"g must be non-positive (inequality feasibility), got max(g) = {g.max().item():.3e}"
                )
            a = torch.sigmoid(g / self.tau)
            lam = a * lam

        return torch.cat([w, nu, lam], dim=1)

    @torch.no_grad()
    def predict(self, p_batch: torch.Tensor) -> torch.Tensor:
        return self(p_batch)

    def save(self, save_dir):
        """Save PredictorTwoStage config and both sub-network weights to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            #"model_params": self.nlp.nmpc.model_params.to_dict(),
            #"nmpc_params": self.nlp.nmpc.nmpc_params.to_dict(),
            "nlp_params": {
                "device": self.nlp.device,
                "dtype": str(self.nlp.dtype).split(".")[-1],
                "eps": self.nlp.eps,
                "sigma": self.nlp.sigma,
                "n_threads": self.nlp.n_threads,
            },
            "nn_w_params": self.model_w.to_dict(),
            "nn_dual_params": self.model_dual.to_dict(),
            "predictor_params": {
                "w_sigmoid": self.w_sigmoid,
                "lam_softplus": self.lam_softplus,
                "use_gate": self.use_gate,
                "tau": self.tau,
                "convexification": self.convexification,
            },
        }

        with open(save_dir / "predictor_two_stage_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model_w.save_weights(save_dir, file_name="weights_w.pt")
        self.model_dual.save_weights(save_dir, file_name="weights_dual.pt")

    @classmethod
    def load(cls, load_dir) -> "PredictorTwoStage":
        """Reconstruct a PredictorTwoStage from a directory written by save()."""
        from nlp_handler import NLPHandler
        from models.cstr_cascade import CSTRParams
        from models.cstr_nmpc import CSTRNMPC, NMPCParams

        load_dir = Path(load_dir)

        with open(load_dir / "predictor_two_stage_config.json") as f:
            config = json.load(f)

        model_params = CSTRParams.from_dict(config["model_params"])
        nmpc_params = NMPCParams.from_dict(config["nmpc_params"])
        nmpc = CSTRNMPC(model_params, nmpc_params)

        nlp_cfg = dict(config["nlp_params"])
        nlp_cfg["dtype"] = getattr(torch, nlp_cfg["dtype"])
        nlp_cfg.pop("batch_size", None)
        nlp = NLPHandler(nmpc, **nlp_cfg)

        nn_w_cfg = dict(config["nn_w_params"])
        nn_w_cfg["nn_dtype"] = getattr(torch, nn_w_cfg["nn_dtype"])
        model_w = FeedforwardNN(**nn_w_cfg)
        model_w.load_weights(load_dir, file_name="weights_w.pt")
        model_w.to(nlp.device)

        nn_dual_cfg = dict(config["nn_dual_params"])
        nn_dual_cfg["nn_dtype"] = getattr(torch, nn_dual_cfg["nn_dtype"])
        model_dual = FeedforwardNN(**nn_dual_cfg)
        model_dual.load_weights(load_dir, file_name="weights_dual.pt")
        model_dual.to(nlp.device)

        return cls(model_w, model_dual, nlp, **config["predictor_params"])


def load_predictor(load_dir,nlp):
    """Load any supported predictor type from a run directory."""
    load_dir = Path(load_dir)

    if (load_dir / "predictor_decomposed_config.json").exists():
        return PredictorDecomposed.load(load_dir)
    if (load_dir / "predictor_two_stage_config.json").exists():
        return PredictorTwoStage.load(load_dir)
    if (load_dir / "predictor_config.json").exists():
        return Predictor.load(load_dir,nlp)

    raise FileNotFoundError(
        f"No predictor config found in {load_dir} "
        "(expected predictor_config.json, predictor_two_stage_config.json, "
        "or predictor_decomposed_config.json)"
    )
