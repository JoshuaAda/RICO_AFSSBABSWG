import math
import copy
import json
import time
from pathlib import Path
from optimization import Optimization_Problem
import torch
import numpy as np
from ..training.training_utils import adaptive_clip_grad
from .nn_models import FeedforwardNN


class Solver(torch.nn.Module):
    def __init__(self,model,nlp,predictor=None,convexification=False,gamma=0.1,jvp_linesearch: bool = False):
        super().__init__()
        self.model = model #step model
        self.nlp = nlp
        self.predictor = predictor
        self.gamma = gamma # solver_cfg
        self.convexification = convexification # solver_cfg
        self.jvp_linesearch = bool(jvp_linesearch)
        self.offset = 1e-16
        self.norm_Fk_max = 1.0
        self._single_eval_ready = False
        self._single_model = None
        self._single_nn_device = None
        self._single_kkt_backend = None
        self._single_kkt_device = None
        self._single_nn_compile_mode = "none"
        self._single_diverge_factor = 10.0
        self._single_alpha_decay = 0.95
        self._single_alpha_growth = 1.0 / 0.95
        self.initialize_functions()
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def initialize_functions(self):
        # Select Tk function based on convexification flag
        if self.convexification:
            raise NotImplementedError("Convexified Tk not yet implemented in NLPHandler")
        # print("Solver functions initialized.")

    def setup_for_batch_eval(self, compile_mode="reduce-overhead"):
        self.model.eval()
        if compile_mode is not None:
            self.model = torch.compile(
                self.model,
                mode=compile_mode,
                fullgraph=True,
            )

    def _get_solver_params_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "convexification": self.convexification,
            "jvp_linesearch": self.jvp_linesearch,
        }

    def _compute_residual_features(self, zk, p):
        """Compute KKT residual features for monolithic solvers.

        Returns:
            Fk_batch: Raw KKT residual, shape (B, n_z).
            Fk_normalized: Residual normalized by its 2-norm, shape (B, n_z).
            norm_Fk: 2-norm of residual + offset, shape (B, 1).
            norm_Fk_log: Log of norm_Fk, shape (B, 1).
        """
        #t1=time.time()
        if self.nlp.device== "cuda":
            Fk_batch = self.nlp.Fk_batch_func(zk, p)
        else:
            Fk_batch=self.nlp.fk(zk,p)#Fk_batch_func(zk, p)
        #t2 = time.time()
        #print("Compute Fk time: " + str(t2 - t1))
        norm_Fk = torch.linalg.vector_norm(Fk_batch, ord=2, dim=1, keepdim=True) + self.offset
        norm_Fk_log = torch.log(norm_Fk)
        Fk_normalized = Fk_batch / norm_Fk
        return Fk_batch, Fk_normalized, norm_Fk, norm_Fk_log
    
    def forward(self,zk,p,apply_jvp: bool | None = None):
        """Predict step direction for given batch of primal-dual iterates and parameters."""
        t1=time.time()
        with torch.no_grad():
            Fk_batch, Fk_normalized, norm_Fk, norm_Fk_log = self._compute_residual_features(zk, p)

        nn_dtype = next(self.model.parameters()).dtype
        nn_inputs_batch = torch.hstack((p, Fk_normalized, norm_Fk_log)).to(dtype=nn_dtype)
        #print(norm_Fk[0])
        norm_Fk_clamped = torch.clamp(norm_Fk, max=self.norm_Fk_max)
        #end=False
        #t2 = time.time()
        #if all(norm_Fk<1e-3):
        #    end=True
        #print("Compute Fk time: " + str(t2 - t1))
        dzk = self.model(nn_inputs_batch)
        dzk = dzk.to(dtype=zk.dtype)
        dzk = dzk * norm_Fk_clamped

        dzk = dzk * self.gamma
        t3 = time.time()
        #print("Compute network time: " + str(t3 - t2))
        if apply_jvp is None:
            apply_jvp = self.jvp_linesearch
        if apply_jvp:
            with torch.no_grad():
                alpha_jvp = self.nlp.alpha_batch_func(zk, p, dzk.detach())
                if alpha_jvp.dim() == 1:
                    alpha_jvp = alpha_jvp.unsqueeze(1)
                alpha_jvp = alpha_jvp.clamp(min=1e-2, max=1.0)
            dzk = dzk * alpha_jvp

        return dzk#,end

    def training_step(self, zk_batch, p_batch, optim, use_adaptive_clip_grad=False):
        # Zero Grads
        optim.zero_grad(set_to_none=True)

        # Predict step
        # dzk_batch = self.forward(zk_batch, p_batch)
        # Compute new iterates
        # zk_new = zk_batch + dzk_batch
        
        # Multi Step
        zk_new = zk_batch
        for i in range(1):
            dzk_batch = self.forward(zk_new, p_batch)
            zk_new = zk_new + dzk_batch

        # Evaluate per-sample loss
        Tk_batch = self.nlp.Tk_batch_func(zk_new, p_batch)

        # Compute loss
        loss = torch.mean(torch.log(Tk_batch + self.offset**2))

        # Backprop
        loss.backward()

        # Gradient clipping
        if use_adaptive_clip_grad:
            adaptive_clip_grad(self.model.parameters(), clip_factor=0.01, eps=1e-3)

        # Step
        optim.step()

        # with torch.no_grad():
        #     zk_new = self.nlp.project_z_box(zk_new)

        return loss, zk_new, Tk_batch

    def sample_z_p(self,n_samples):
        p_batch =self.nlp.batch_gen_p(n_samples,offset=0.0)#torch.Tensor(np.repeat(np.array([-2,2,1,-1]),256).reshape((4,256)).T).to("cuda").double()# self.nlp.batch_gen_p(n_samples,offset=0.0)
        if self.predictor is None:
            zk_batch = self.nlp.batch_gen_z(n_samples)
        else:
            zk_batch = self.predictor.predict(p_batch).to(self.nlp.dtype)
            # raise NotImplementedError("Resampling with predictor not implemented yet.")
        return zk_batch, p_batch

    def _as_numpy_1d(self, x, expected_size: int, name: str) -> np.ndarray:
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        x_np = np.asarray(x_np, dtype=np.float64).reshape(-1)
        if x_np.shape[0] != int(expected_size):
            raise ValueError(
                f"{name} must have size {expected_size}, got {x_np.shape[0]}"
            )
        return x_np

    def _as_torch_1d(
        self,
        x,
        expected_size: int,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            x_t = x.detach().to(device=device, dtype=dtype)
        else:
            x_t = torch.tensor(x, device=device, dtype=dtype)
        x_t = x_t.reshape(-1)
        if x_t.shape[0] != int(expected_size):
            raise ValueError(
                f"{name} must have size {expected_size}, got {x_t.shape[0]}"
            )
        return x_t

    def _clone_state(self, z):
        if torch.is_tensor(z):
            return z.detach().clone()
        return np.array(z, dtype=np.float64, copy=True)

    def _state_axpy(self, z, dz, alpha: float):
        if torch.is_tensor(z):
            return z + float(alpha) * dz
        return z + float(alpha) * dz

    def _fk_inf_from_residual(self, fk) -> float:
        if torch.is_tensor(fk):
            return float(torch.linalg.vector_norm(fk, ord=float("inf")).item())
        return float(np.max(np.abs(fk)))

    def _to_output_tensor(self, z):
        if torch.is_tensor(z):
            return z.detach().clone()
        return torch.from_numpy(np.asarray(z, dtype=np.float64)).to(dtype=self.nlp.dtype)

    def _setup_single_eval_cache(self) -> None:
        """Hook for subclasses to prepare additional cached tensors."""
        return

    def _ensure_single_eval_ready(self) -> None:
        if not self._single_eval_ready:
            raise RuntimeError(
                "Single-eval fast path not set up. Call setup_for_single_eval(...) first."
            )

    def setup_for_single_eval(
        self,
        nn_device: str | None = None,
        kkt_backend: str = "casadi_single",
        kkt_device: str = "cpu",
        nn_compile_mode: str = "none",
        batch_size: int = 1
    ) -> None:
        """Prepare a fast single-sample solve path.

        This path is intentionally minimal:
        - no warm-start/predictor logic
        - line-search only
        - no projection/clipping
        """
        if nn_device is None:
            nn_device = "cuda" if torch.cuda.is_available() else "cpu"

        nn_device = str(nn_device)
        kkt_backend = str(kkt_backend)
        nn_compile_mode = str(nn_compile_mode)

        if kkt_backend not in ("casadi_single", "cusadi_batch"):
            raise ValueError(
                f"Unsupported kkt_backend '{kkt_backend}'. "
                "Expected one of: 'casadi_single', 'cusadi_batch'."
            )
        if nn_compile_mode not in ("none", "compile", "jit"):
            raise ValueError(
                f"Unsupported nn_compile_mode '{nn_compile_mode}'. "
                "Expected one of: 'none', 'compile', 'jit'."
            )

        if kkt_backend == "casadi_single":
            kkt_device = "cpu"
        else:
            kkt_device = str(kkt_device)
            if not kkt_device.startswith("cuda"):
                raise ValueError(
                    "kkt_backend='cusadi_batch' requires kkt_device to be CUDA."
                )
            if not str(self.nlp.device).startswith("cuda"):
                raise ValueError(
                    "kkt_backend='cusadi_batch' requires nlp.device to be CUDA."
                )
            self.nlp.setup_for_batch(batch_size=batch_size, force_rebuild=False)

        model_fast = copy.deepcopy(self.model)
        model_fast = model_fast.to(device=nn_device, dtype=torch.float32)
        model_fast.eval()

        if nn_compile_mode == "compile":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch version.")
            model_fast = torch.compile(model_fast, mode="reduce-overhead")
        elif nn_compile_mode == "jit":
            model_fast = torch.jit.script(model_fast)
            model_fast = torch.jit.optimize_for_inference(model_fast)

        self._single_model = model_fast
        self._single_nn_device = torch.device(nn_device)
        self._single_kkt_backend = kkt_backend
        self._single_kkt_device = torch.device(kkt_device)
        self._single_nn_compile_mode = nn_compile_mode
        self._setup_single_eval_cache()
        self._single_eval_ready = True

    def setup_for_gpu_batch_eval(
            self,
            nn_device: str | None = None,
            kkt_backend: str = "casadi_single",
            kkt_device: str = "cpu",
            nn_compile_mode: str = "none",
    ) -> None:
        """Prepare a fast single-sample solve path.

        This path is intentionally minimal:
        - no warm-start/predictor logic
        - line-search only
        - no projection/clipping
        """
        if nn_device is None:
            nn_device = "cuda" if torch.cuda.is_available() else "cpu"

        nn_device = str(nn_device)
        kkt_backend = str(kkt_backend)
        nn_compile_mode = str(nn_compile_mode)

        if kkt_backend not in ("casadi_single", "cusadi_batch"):
            raise ValueError(
                f"Unsupported kkt_backend '{kkt_backend}'. "
                "Expected one of: 'casadi_single', 'cusadi_batch'."
            )
        if nn_compile_mode not in ("none", "compile", "jit"):
            raise ValueError(
                f"Unsupported nn_compile_mode '{nn_compile_mode}'. "
                "Expected one of: 'none', 'compile', 'jit'."
            )

        if kkt_backend == "casadi_single":
            kkt_device = "cpu"
        else:
            kkt_device = str(kkt_device)
            if not kkt_device.startswith("cuda"):
                raise ValueError(
                    "kkt_backend='cusadi_batch' requires kkt_device to be CUDA."
                )
            if not str(self.nlp.device).startswith("cuda"):
                raise ValueError(
                    "kkt_backend='cusadi_batch' requires nlp.device to be CUDA."
                )
            self.nlp.setup_for_batch(batch_size=1, force_rebuild=False)

        model_fast = copy.deepcopy(self.model)
        model_fast = model_fast.to(device=nn_device, dtype=torch.float32)
        model_fast.eval()

        if nn_compile_mode == "compile":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch version.")
            model_fast = torch.compile(model_fast, mode="reduce-overhead")
        elif nn_compile_mode == "jit":
            model_fast = torch.jit.script(model_fast)
            model_fast = torch.jit.optimize_for_inference(model_fast)

        self._single_model = model_fast
        self._single_nn_device = torch.device(nn_device)
        self._single_kkt_backend = kkt_backend
        self._single_kkt_device = torch.device(kkt_device)
        self._single_nn_compile_mode = nn_compile_mode
        self._setup_single_eval_cache()
        self._single_eval_ready = True

    def _eval_fk_single(self, z, p):
        self._ensure_single_eval_ready()
        if self._single_kkt_backend == "casadi_single":
            return np.asarray(self.nlp.Fk_func_ca(z, p), dtype=np.float64).reshape(-1)
        if self._single_kkt_backend == "cusadi_batch":
            return self.nlp.Fk_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    def _eval_kkt_single(self, z, p):
        self._ensure_single_eval_ready()
        if self._single_kkt_backend == "casadi_single":
            return np.asarray(self.nlp.KKT_func_ca(z, p), dtype=np.float64).reshape(-1)
        if self._single_kkt_backend == "cusadi_batch":
            return self.nlp.KKT_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    def _eval_tk_single(self, z, p) -> float:
        self._ensure_single_eval_ready()
        if self._single_kkt_backend == "casadi_single":
            if hasattr(self.nlp, "Tk_nodiff_func_ca"):
                tk_raw = self.nlp.Tk_nodiff_func_ca(z, p)
            elif hasattr(self.nlp, "Tk_nodiff_reg_func_ca"):
                tk_raw = self.nlp.Tk_nodiff_reg_func_ca(z, p)
            else:
                raise AttributeError(
                    "Could not find Tk_nodiff CasADi function on NLP handler."
                )
            return float(np.asarray(tk_raw, dtype=np.float64).reshape(-1)[0])
        if self._single_kkt_backend == "cusadi_batch":
            if hasattr(self.nlp, "Tk_batch_nodiff_func"):
                tk_t = self.nlp.Tk_batch_nodiff_func(z.unsqueeze(0), p.unsqueeze(0))
            else:
                tk_t = self.nlp.Tk_batch_func(z.unsqueeze(0), p.unsqueeze(0))
            return float(tk_t.reshape(-1)[0].item())
        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    @torch.no_grad()
    def _forward_fast_single(self, z, p):
        self._ensure_single_eval_ready()
        fk = self._eval_fk_single(z, p)

        if self._single_kkt_backend == "casadi_single":
            norm_fk = float(np.linalg.norm(fk, ord=2) + self.offset)
            fk_normalized = fk / norm_fk
            nn_input_np = np.concatenate(
                [p, fk_normalized, np.array([math.log(norm_fk)], dtype=np.float64)]
            )
            nn_input = torch.from_numpy(nn_input_np).to(
                device=self._single_nn_device, dtype=torch.float32
            )
            nn_input = nn_input.unsqueeze(0)
            norm_fk_clamped = min(norm_fk, self.norm_Fk_max)
            dz = self._single_model(nn_input).squeeze(0)
            dz = dz * float(norm_fk_clamped)
            dz = dz * float(self.gamma)
            return dz.detach().cpu().numpy().astype(np.float64, copy=False)

        if self._single_kkt_backend == "cusadi_batch":
            norm_fk = torch.linalg.vector_norm(fk, ord=2,dim=1) + self.offset
            fk_normalized = fk / norm_fk.reshape((-1,1))
            norm_fk_log = torch.log(norm_fk).reshape((-1,1))#.reshape(1)
            nn_input = torch.cat([p, fk_normalized, norm_fk_log], dim=1)
            nn_input = nn_input.to(device=self._single_nn_device, dtype=torch.float32)
            nn_input = nn_input.unsqueeze(0)
            #norm_fk_clamped = min((norm_fk.item()), self.norm_Fk_max)
            dz = self._single_model(nn_input).squeeze(0)
            dz = norm_fk.float().reshape((-1,1))*dz#dz * norm_fk#_clamped)
            dz = dz * float(self.gamma)
            return dz.to(device=self._single_kkt_device, dtype=self.nlp.dtype)

        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    @torch.no_grad()
    def solve_fast_single(
        self,
        p,
        z0,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 1.0,
        update_mode: str = "line_search",
    ):
        """Fast single-sample solve with best-so-far line search.

        Returns:
            (z_best, n_iter, success, fk_inf_best)
        """
        self._ensure_single_eval_ready()

        if update_mode != "line_search":
            raise ValueError("solve_fast_single only supports update_mode='line_search'.")
        if z0 is None:
            raise ValueError("z0 is mandatory for solve_fast_single.")

        if self._single_kkt_backend == "casadi_single":
            p_state = self._as_numpy_1d(p, self.nlp.n_p, "p")
            z_state = self._as_numpy_1d(z0, self.nlp.n_z, "z0")
        else:
            p_state = self._as_torch_1d(
                p,
                self.nlp.n_p,
                "p",
                device=self._single_kkt_device,
                dtype=self.nlp.dtype,
            )
            z_state = self._as_torch_1d(
                z0,
                self.nlp.n_z,
                "z0",
                device=self._single_kkt_device,
                dtype=self.nlp.dtype,
            )

        z_best = self._clone_state(z_state)
        fk_best = self._eval_fk_single(z_best, p_state)
        fk_inf_best = self._fk_inf_from_residual(fk_best)

        if fk_inf_best < float(tol):
            return self._to_output_tensor(z_best), 0, True, float(fk_inf_best)

        alpha_k = float(alpha)
        success = False
        n_iter = 0

        for i in range(int(max_iter)):
            dz = self._forward_fast_single(z_state, p_state)
            z_candidate = self._state_axpy(z_state, dz, alpha_k)

            fk_candidate = self._eval_fk_single(z_candidate, p_state)
            fk_inf_candidate = self._fk_inf_from_residual(fk_candidate)

            if fk_inf_candidate < fk_inf_best:
                z_best = self._clone_state(z_candidate)
                fk_inf_best = fk_inf_candidate
                alpha_k = min(float(alpha), alpha_k * self._single_alpha_growth)

            if fk_inf_candidate > self._single_diverge_factor * fk_inf_best:
                z_state = self._clone_state(z_best)
                alpha_k = alpha_k * self._single_alpha_decay
            else:
                z_state = z_candidate

            n_iter = i + 1
            if fk_inf_best < float(tol):
                success = True
                break

        return self._to_output_tensor(z_best), n_iter, success, float(fk_inf_best)

    @torch.no_grad()
    def solve_fast_gpu(
            self,
            p,
            z0,
            max_iter: int = 100,
            tol: float = 1e-6,
            alpha: float = 1.0,
            update_mode: str = "line_search",
    ):
        """Fast single-sample solve with best-so-far line search.

        Returns:
            (z_best, n_iter, success, fk_inf_best)
        """
        self._ensure_single_eval_ready()

        if update_mode != "line_search":
            raise ValueError("solve_fast_single only supports update_mode='line_search'.")
        if z0 is None:
            raise ValueError("z0 is mandatory for solve_fast_single.")

        if self._single_kkt_backend == "casadi_single":
            p_state = self._as_numpy_1d(p, self.nlp.n_p, "p")
            z_state = self._as_numpy_1d(z0, self.nlp.n_z, "z0")
        else:
            """
            p_state = self._as_torch_1d(
                p,
                self.nlp.n_p,
                "p",
                device=self._single_kkt_device,
                dtype=self.nlp.dtype,
            )
            z_state = self._as_torch_1d(
                z0,
                self.nlp.n_z,
                "z0",
                device=self._single_kkt_device,
                dtype=self.nlp.dtype,
            )
            """
            z_state=z0
            p_state=p
        z_best = self._clone_state(z_state)
        fk_best = self._eval_fk_single(z_best, p_state)
        fk_inf_best = self._fk_inf_from_residual(fk_best)

        if fk_inf_best < float(tol):
            return self._to_output_tensor(z_best), 0, True, float(fk_inf_best)

        alpha_k = float(alpha)
        success = False
        n_iter = 0

        for i in range(int(max_iter)):
            dz = self._forward_fast_single(z_state, p_state)
            z_candidate = self._state_axpy(z_state, dz, alpha_k)

            fk_candidate = self._eval_fk_single(z_candidate, p_state)
            fk_inf_candidate = self._fk_inf_from_residual(fk_candidate)

            if fk_inf_candidate < fk_inf_best:
                z_best = self._clone_state(z_candidate)
                fk_inf_best = fk_inf_candidate
                alpha_k = min(float(alpha), alpha_k * self._single_alpha_growth)

            if fk_inf_candidate > self._single_diverge_factor * fk_inf_best:
                z_state = self._clone_state(z_best)
                alpha_k = alpha_k * self._single_alpha_decay
            else:
                z_state = z_candidate

            n_iter = i + 1
            if fk_inf_best < float(tol):
                success = True
                break

        return self._to_output_tensor(z_best), n_iter, success, float(fk_inf_best)
    # def train(self, train_config=None, print_frequency=None, log_values=None):
    #     # Preserve nn.Module.train(mode) behaviour
    #     if isinstance(train_config, bool) or train_config is None:
    #         return super().train(train_config if train_config is not None else True)

    #     from src.lisco.training.train_solver import SolverTrainConfig, train_solver

    #     if isinstance(train_config, dict):
    #         config = SolverTrainConfig(**train_config)
    #     else:
    #         config = train_config
    #     return train_solver(self, config, print_frequency=print_frequency, log_values=log_values)

    @torch.no_grad()
    def solve_batch(
        self,
        p_batch,
        max_iter=100,
        alpha=1.0,
        predictor=None,
        update_mode="full",
        return_trajectory=False,
        z0_batch=None,
            armijo_sigma=1e-4,
            armijo_beta=0.7,
            descent_eta=1e-4,
            direction_kappa=100.0,
            max_backtracks=5,
    ):
        Tk_batch_func = self.nlp.Tk_batch_func
        step_func = self.forward

        alpha_batch = torch.ones(p_batch.shape[0],device=p_batch.device)*alpha
        # initial batch
        n_batch = p_batch.shape[0]
        step_time_list = []
        if z0_batch is not None:
            zk_batch = z0_batch.to(device=p_batch.device, dtype=p_batch.dtype).detach().clone()
            #if zk_batch.shape != (n_batch, self.nlp.n_z):
            #    raise ValueError(
            #        f"z0_batch must have shape ({n_batch}, {self.nlp.n_z}), got {tuple(zk_batch.shape)}"
            #    )
        elif predictor is None:
            zk_batch = self.nlp.batch_gen_z(n_batch)
        else:
            pred_tic = time.perf_counter()
            zk_batch = predictor.predict(p_batch).to(p_batch.dtype)
            pred_toc = time.perf_counter()
            pred_time = pred_toc - pred_tic
            step_time_list.append(pred_time)
            #print(np.array(zk_batch[0, 0].detach().cpu()))
            #print(np.array(zk_batch[0, 80].detach().cpu()))
            #print(np.array(zk_batch[0, 120].detach().cpu()))

        # zk_batch = self.nlp.project_z_box(zk_batch)

        if update_mode in ["line_search", "armijo"]:
            T0_batch = Tk_batch_func(zk_batch,p_batch)
            zk_best = zk_batch.detach().clone()
            Tk_best = T0_batch.detach().clone()
        if update_mode == "safeguarded_armijo":
            Tk_current = (
                self.nlp.Tk_batch_nodiff_func(zk_batch, p_batch)
                .reshape(n_batch)
            )
        # Pre-allocate trajectory buffer if needed (GPU-friendly)
        if return_trajectory:
            zk_traj_buffer = torch.empty((max_iter + 1, n_batch, zk_batch.shape[1]),
                                         dtype=zk_batch.dtype, device=zk_batch.device)
            zk_traj_buffer[0] = zk_batch.detach().clone()

        # LOOP
        for i in range(max_iter):
            #print(f"Solve iteration {i+1}/{max_iter}")
            step_tic = time.perf_counter()

            dzk_batch = self.forward(zk_batch, p_batch, apply_jvp=False)

            alpha_eff = alpha_batch
            step_toc = time.perf_counter()
            if self.jvp_linesearch:
                alpha_jvp = self.nlp.alpha_batch_func(zk_batch, p_batch, dzk_batch).squeeze(-1)
                alpha_jvp = alpha_jvp.clamp(min=1e-2, max=1.0)
                alpha_eff = torch.minimum(alpha_batch, alpha_jvp)

            if update_mode == "full":
                zk_batch = zk_batch + alpha_eff[:,None]*dzk_batch
                # zk_batch = self.nlp.project_z_box(zk_batch)
                #print(np.array(zk_batch[0,0].detach().cpu()))
                #print(np.array(zk_batch[0, 80].detach().cpu()))
                #print(np.array(zk_batch[0, 120].detach().cpu()))
            elif update_mode == "line_search":
                # update
                zk_batch = zk_batch + alpha_eff[:,None]*dzk_batch
                # zk_batch = self.nlp.project_z_box(zk_batch)
                Tk_batch = Tk_batch_func(zk_batch,p_batch)
                # update best iterates
                mask_best = Tk_batch < Tk_best
                zk_best[mask_best] = zk_batch[mask_best].detach().clone()
                Tk_best[mask_best] = Tk_batch[mask_best].detach().clone()
                # diverging
                mask_diverged = Tk_batch > 10*Tk_best
                # reset to best + reduce alpha
                zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()
                alpha_batch[mask_diverged] = alpha_batch[mask_diverged]*0.95
                alpha_batch[mask_best] = torch.clip(alpha_batch[mask_best]/0.95,0.0,alpha)


            elif update_mode == "safeguarded_armijo":

                grad_Tk = (
                    self.nlp.grad_Tk_z_batch_func(zk_batch, p_batch)
                    .reshape(n_batch, -1)
                )

                grad_norm_sq = torch.sum(grad_Tk.square(), dim=1)
                grad_norm = torch.sqrt(grad_norm_sq)

                learned_direction = dzk_batch
                direction_norm = torch.linalg.vector_norm(
                    learned_direction,
                    dim=1,
                )

                eps_machine = torch.finfo(zk_batch.dtype).eps

                # Bound the learned direction:
                # ||d|| <= direction_kappa ||grad Tk||.
                direction_scale = torch.clamp(
                    direction_kappa * grad_norm
                    / (direction_norm + eps_machine),
                    max=1.0,
                )

                learned_direction = (
                        learned_direction * direction_scale[:, None]
                )

                learned_gtd = torch.sum(
                    grad_Tk * learned_direction,
                    dim=1,
                )

                use_learned = (
                        torch.isfinite(learned_direction).all(dim=1)
                        & torch.isfinite(learned_gtd)
                        & (
                                learned_gtd
                                <= -descent_eta * grad_norm_sq
                        )
                )

                direction = torch.where(
                    use_learned[:, None],
                    learned_direction,
                    -grad_Tk,
                )

                directional_derivative = torch.sum(
                    grad_Tk * direction,
                    dim=1,
                )
                d = 0
                for k in range(max_backtracks):

                    candidate = (
                            zk_batch
                            + alpha_batch[:, None] * direction
                    )

                    Tk_candidate = (
                        self.nlp.Tk_batch_nodiff_func(candidate, p_batch)
                        .reshape(n_batch)
                    )

                    armijo_rhs = (
                            Tk_current
                            + armijo_sigma
                            * alpha_batch
                            * directional_derivative
                    )

                    accepted = (
                            torch.isfinite(Tk_candidate)
                            & (Tk_candidate <= armijo_rhs)
                    )

                    # Accept successful samples; rejected samples stay unchanged.
                    zk_batch = torch.where(
                        accepted[:, None],
                        candidate,
                        zk_batch,
                    )

                    Tk_current = torch.where(
                        accepted,
                        Tk_candidate,
                        Tk_current,
                    )
                    if all(accepted):

                        break
                    else:
                        alpha_batch[~accepted]=alpha_batch[~accepted]*armijo_beta

                        #d=d+1
                        #print(d)
                    # Restore the nominal step after success and backtrack after failure.
                alpha_batch = torch.where(
                    accepted,
                    torch.full_like(alpha_batch, float(alpha)),
                    armijo_beta * alpha_batch,
                )
            elif update_mode == "armijo":
                # update
                grad_Tk_z_batch = self.nlp.grad_Tk_z_batch_func(zk_batch,p_batch)
                zk_prev = zk_batch
                zk_batch = zk_batch + alpha_eff[:,None]*dzk_batch
                # zk_batch = self.nlp.project_z_box(zk_batch)
                Tk_batch = Tk_batch_func(zk_batch,p_batch)
                dzk_actual = zk_batch - zk_prev
                armijo_cond = Tk_batch <= Tk_best + 1e-4*torch.sum(grad_Tk_z_batch*dzk_actual,dim=1)
                # update best iterates
                zk_best[armijo_cond] = zk_batch[armijo_cond].detach().clone()
                Tk_best[armijo_cond] = Tk_batch[armijo_cond].detach().clone()
                # diverging
                mask_diverged = ~armijo_cond
                # reset to best + reduce alpha
                zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()
                alpha_batch[mask_diverged] = alpha_batch[mask_diverged]*0.95
                alpha_batch[armijo_cond] = alpha # torch.clip(alpha_batch[armijo_cond]/0.95,0.0,alpha)
            else:
                raise ValueError("Update mode not implemented.")


            step_time = step_toc - step_tic
            step_time_list.append(step_time)

            if return_trajectory:
                if update_mode == "line_search":
                    zk_traj_buffer[i+1] = zk_best.detach().clone()
                else:
                    zk_traj_buffer[i+1] = zk_batch.detach().clone()
            #if end:
            #    break

        # END OF SOLVE
        if update_mode == "line_search":
            zk_batch = zk_best.detach().clone()

        if return_trajectory:
            # Convert buffer to list only after loop completes
            zk_traj = [zk_traj_buffer[j] for j in range(max_iter + 1)]
            return zk_batch, zk_traj, step_time_list
        else:
            return zk_batch, step_time_list

    def get_casadi_solver_step_funcs(self,nlp_casadi,jit=True):
        model = copy.deepcopy(self.model)
        model.cpu()

        if jit:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)

        step_func = nlp_casadi._build_solver_step(model)
        Tk_func = nlp_casadi.Tk_func
        # return nlp_casadi._build_solver_step(model)
        return step_func, Tk_func

    @torch.no_grad()
    def solve_fast_single_cpu(self,solve_step_func,Tk_func,p,predictor=None,alpha=1.0,max_iter=100,tol=1e-6,update_mode="line_search"):
        success = False
        gamma = self.gamma
        device = "cpu"

        if predictor is None:
            zk = self.nlp.batch_gen_z(1).squeeze().to(device).numpy()
        else:
            zk = predictor(torch.from_numpy(p)).squeeze().to(device).numpy()

        # Initializaton for "line_search"
        z_best = zk
        T_best = Tk_func(zk,p)

        if update_mode == "line_search":
            alpha_k = alpha
            for i in range(max_iter):
                dzk, Fk_2_norm = solve_step_func(zk,p)

                if Fk_2_norm < tol:
                    success = True
                    break

                alpha_eff = alpha_k
                if self.jvp_linesearch:
                    alpha_jvp = float(self.nlp.alpha_func_ca(zk, p, dzk * gamma))
                    alpha_jvp = float(np.clip(alpha_jvp, 1e-8, 1.0))
                    alpha_eff = min(alpha_k, alpha_jvp)

                zk = zk + dzk*alpha_eff*gamma
                n_w = int(self.nlp.n_w)
                n_h = int(self.nlp.n_h)
                zk[:n_w] = np.clip(zk[:n_w], 0.0, 1.0)
                zk[n_w + n_h :] = np.clip(zk[n_w + n_h :], 0.0, np.inf)
                Tk = Tk_func(zk,p)

                if Tk < T_best:
                    # update best iterate and error
                    z_best = zk
                    T_best = Tk
                    # increase alpha
                    alpha_k = np.clip(alpha_k/0.95,0.0,alpha) # two-way backtracking line search, increase alpha again if Tk is smaller than Tk_best (not larger than initial alpha)

                if Tk > 10*T_best:
                    zk = z_best
                    # decrease alpha
                    alpha_k = alpha_k*0.95

        else:
            raise ValueError("Update mode not implemented.")

        return torch.from_numpy(zk),i+1,success

    def save(self, save_dir):
        """Save solver config (JSON) and model weights (.pt) to save_dir.

        Writes:
            solver_config.json  — all hyperparameters, human-readable
            weights.pt          — FeedforwardNN state dict
            predictor/          — subdirectory written by Predictor.save(),
                                  only present when self.predictor is not None

        The full Solver can be reconstructed with Solver.load(save_dir)
        (or the appropriate subclass, e.g. SolverDecomposed.load(save_dir)).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "solver_class":  type(self).__name__,
            #"model_params":  self.nlp.nmpc.model_params.to_dict(),
            #"nmpc_params":   self.nlp.nmpc.nmpc_params.to_dict(),
            "nlp_params": {
                "device":     self.nlp.device,
                "dtype":      str(self.nlp.dtype).split(".")[-1],
                "eps":        self.nlp.eps,
                "sigma":      self.nlp.sigma,
                "n_threads":  self.nlp.n_threads,
                "distributed_KKT_functions": self.nlp.distributed_KKT_functions,
            },
            "nn_params":    self.model.to_dict(),
            "solver_params": self._get_solver_params_dict(),
            "has_predictor": self.predictor is not None,
        }

        with open(save_dir / "solver_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(save_dir)

        if self.predictor is not None:
            self.predictor.save(save_dir / "predictor")

    @classmethod
    def load(cls, load_dir,nlp) -> "Solver":
        """Reconstruct a Solver from a directory written by save().

        Rebuilds the full chain: CSTRParams → NMPCParams → CSTRNMPC →
        NLPHandler → FeedforwardNN (with loaded weights) → Solver.

        Call on the correct subclass to get the right type back, e.g.:
            SolverDecomposed.load(path)
        A ValueError is raised if the stored solver_class does not match cls.

        Note: hardcoded to CSTRNMPC; generalise if other NMPC types are added.
        """
        #from nlp_handler import NLPHandler
        from .predictor import load_predictor
        #from ...models.cstr_cascade import CSTRParams
        #from ...models.cstr_nmpc import CSTRNMPC, NMPCParams
        #nmpc = Optimization_Problem()
        load_dir = Path(load_dir)

        with open(load_dir / "solver_config.json") as f:
            config = json.load(f)

        if cls.__name__ != config["solver_class"]:
            raise ValueError(
                f"Class mismatch: called {cls.__name__}.load() but "
                f"saved solver_class is '{config['solver_class']}'"
            )

        #model_params = CSTRParams.from_dict(config["model_params"])
        #nmpc_params  = NMPCParams.from_dict(config["nmpc_params"])
        #nmpc = CSTRNMPC(model_params, nmpc_params)

        nlp_cfg = dict(config["nlp_params"])
        nlp_cfg["dtype"] = getattr(torch, nlp_cfg["dtype"])
        nlp_cfg.pop("batch_size", None)   # compat: old files may contain this key
        #nlp = NLPHandler(nmpc, **nlp_cfg)

        solver_params = dict(config.get("solver_params", {}))
        if cls.__name__ == "SolverDecomposed":
            required = ("n_nodes", "n_slide_steps", "overlap_merge_mode")
            missing = [key for key in required if key not in solver_params]
            if missing:
                raise ValueError(
                    "Checkpoint is missing decomposed solver parameters "
                    f"{missing}. This checkpoint was likely saved before "
                    "SolverDecomposed-specific serialization was added."
                )

        nn_cfg = dict(config["nn_params"])
        nn_cfg["nn_dtype"] = getattr(torch, nn_cfg["nn_dtype"])
        model = FeedforwardNN(**nn_cfg)
        model.load_weights(load_dir)
        model.to(nlp.device)

        predictor = load_predictor(load_dir / "predictor",nlp) if config["has_predictor"] else None

        return cls(model, nlp, predictor=predictor, **solver_params)


class SolverDecomposed(Solver):
    """Windowed decomposed solver with configurable overlap and merge strategy.

    The shared NN is evaluated in parallel over sliding windows of `n_nodes`
    stages. Consecutive windows advance by `n_slide_steps`.

    NN input per window:
        [p,
         Fk_stages_normalized (covered stages, flattened),
         log(||Fk_stage||) (covered stages, flattened),
         positional encoding per covered stage (flattened)]

    NN output per window:
        [dz_stage_0, ..., dz_stage_{n_nodes-1}] (flattened)

    Overlapping stage predictions are merged using `overlap_merge_mode` and then
    scaled by the clamped per-stage residual norm.
    """

    def __init__(
        self,
        model,
        nlp,
        n_nodes: int,
        n_slide_steps: int,
        overlap_merge_mode: str = "average",
        predictor=None,
        convexification: bool = False,
        gamma: float = 0.1,
        jvp_linesearch: bool = False,
    ):
        self.n_nodes = int(n_nodes)
        self.n_slide_steps = int(n_slide_steps)
        self.overlap_merge_mode = str(overlap_merge_mode)

        super().__init__(
            model=model,
            nlp=nlp,
            predictor=predictor,
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )

        self._validate_config(self.nlp.N)
        self.n_windows = 1 + (self.nlp.N - self.n_nodes) // self.n_slide_steps
        starts = torch.arange(self.n_windows, device=self.nlp.device, dtype=torch.long) * self.n_slide_steps
        offsets = torch.arange(self.n_nodes, device=self.nlp.device, dtype=torch.long)
        self._window_stage_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        self._window_stage_indices_flat = self._window_stage_indices.reshape(-1)
        self._avg_merge_counts = torch.bincount(
            self._window_stage_indices_flat,
            minlength=self.nlp.N,
        )

        if self.overlap_merge_mode == "handoff":
            owner_w, owner_j = self._build_handoff_owner_map()
            self._handoff_owner_window_idx = owner_w
            self._handoff_owner_local_idx = owner_j
        else:
            self._handoff_owner_window_idx = None
            self._handoff_owner_local_idx = None

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
        if self.overlap_merge_mode == "handoff":
            if (self.n_nodes % 2) != (self.n_slide_steps % 2):
                raise ValueError(
                    "handoff mode requires n_nodes and n_slide_steps to have the same parity "
                    "(both even or both odd) to avoid center-tie cases."
                )

    def _build_handoff_owner_map(self) -> tuple[list[int], list[int]]:
        """Build deterministic ownership map for merge mode='handoff'."""
        N = self.nlp.N
        m = self.n_nodes
        s = self.n_slide_steps
        center_offset = 0.5 * (m - 1)

        owner_window_idx: list[int] = []
        owner_local_idx: list[int] = []

        for k in range(N):
            w = math.floor(((float(k) - center_offset) / s) + 0.5)
            w = min(max(w, 0), self.n_windows - 1)
            owner_window_idx.append(w)
            owner_local_idx.append(k - w * s)

        return owner_window_idx, owner_local_idx

    def _get_solver_params_dict(self) -> dict:
        params = super()._get_solver_params_dict()
        params.update(
            {
                "n_nodes": self.n_nodes,
                "n_slide_steps": self.n_slide_steps,
                "overlap_merge_mode": self.overlap_merge_mode,
            }
        )
        return params

    def setup_for_single_eval(
        self,
        nn_device: str | None = None,
        kkt_backend: str = "casadi_single",
        kkt_device: str = "cpu",
        nn_compile_mode: str = "none",
        batch_size: int = 1
    ) -> None:
        if self.overlap_merge_mode == "handoff":
            raise ValueError(
                "setup_for_single_eval does not support overlap_merge_mode='handoff'."
            )
        super().setup_for_single_eval(
            nn_device=nn_device,
            kkt_backend=kkt_backend,
            kkt_device=kkt_device,
            nn_compile_mode=nn_compile_mode,
        )

    def _setup_single_eval_cache(self) -> None:
        self._single_window_stage_indices = self._window_stage_indices.to(
            device=self._single_nn_device
        )
        self._single_window_stage_indices_flat = self._window_stage_indices_flat.to(
            device=self._single_nn_device
        )
        self._single_avg_merge_counts = self._avg_merge_counts.to(
            device=self._single_nn_device, dtype=torch.float32
        )

        if self.nlp.N == 1:
            pos_all = torch.tensor(
                [[0.0, 1.0]], device=self._single_nn_device, dtype=torch.float32
            )
        else:
            k = torch.arange(self.nlp.N, device=self._single_nn_device, dtype=torch.float32).view(
                self.nlp.N, 1
            )
            pos_forward = k / float(self.nlp.N - 1)
            pos_backward = 1.0 - pos_forward
            pos_all = torch.cat([pos_forward, pos_backward], dim=1)

        pos_windows = pos_all[self._single_window_stage_indices]
        self._single_pos_windows_flat = pos_windows.reshape(self.n_windows, 2 * self.n_nodes)
    
    def _compute_stage_residual_features(self, zk, p, compute_global_norm=False, clamp_log=None):
        """Compute KKT residual features decomposed into MPC stages.

        Args:
            zk: Current primal-dual variables, shape (B, n_z).
            p: Parameters, shape (B, n_p).
            compute_global_norm: Whether to also compute global (non-decomposed) norm.
            clamp_log: Optional (min, max) tuple to clamp log-norm values.

        Returns:
            Fk_batch: Raw full KKT residual, shape (B, n_z).
            Fk_stages: Decomposed residual, shape (B, N, n_z_stage).
            Fk_stages_normalized: Per-stage normalized residual, shape (B, N, n_z_stage).
            norm_Fk_stages: Per-stage 2-norm + offset, shape (B, N, 1).
            norm_Fk_stages_log: Log of per-stage norm, shape (B, N, 1).
            norm_Fk_global: Global 2-norm + offset, shape (B, 1) or None.
            norm_Fk_global_log: Log of global norm, shape (B, 1) or None.
        """
        Fk_batch = self.nlp.Fk_batch_func(zk, p)
        Fk_stages = self.nlp.decompose_Fk_batch(Fk_batch)

        norm_Fk_stages = torch.linalg.vector_norm(Fk_stages, ord=2, dim=-1, keepdim=True) + self.offset
        norm_Fk_stages_log = torch.log(norm_Fk_stages)
        if clamp_log is not None:
            norm_Fk_stages_log = norm_Fk_stages_log.clamp(*clamp_log)
        Fk_stages_normalized = Fk_stages / norm_Fk_stages

        norm_Fk_global = norm_Fk_global_log = None
        if compute_global_norm:
            norm_Fk_global = torch.linalg.vector_norm(Fk_batch, ord=2, dim=-1, keepdim=True) + self.offset
            norm_Fk_global_log = torch.log(norm_Fk_global)
            if clamp_log is not None:
                norm_Fk_global_log = norm_Fk_global_log.clamp(*clamp_log)

        return Fk_batch, Fk_stages, Fk_stages_normalized, norm_Fk_stages, norm_Fk_stages_log, norm_Fk_global, norm_Fk_global_log

    def _map_window_inputs(self, zk: torch.Tensor, p: torch.Tensor):
        B = zk.shape[0]
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage

        (
            _Fk_batch,
            _Fk_stages,
            Fk_stages_normalized,
            norm_Fk_stages,
            norm_Fk_stages_log,
            _norm_Fk_global,
            _norm_Fk_global_log,
        ) = self._compute_stage_residual_features(zk, p, compute_global_norm=False)

        window_stage_indices = self._window_stage_indices
        n_windows = self.n_windows

        Fk_windows = Fk_stages_normalized[:, window_stage_indices, :]
        norm_windows = norm_Fk_stages_log[:, window_stage_indices, :]

        Fk_windows_flat = Fk_windows.reshape(B, n_windows, self.n_nodes * n_z_stage)
        norm_windows_flat = norm_windows.reshape(B, n_windows, self.n_nodes)

        if N == 1:
            pos_all = torch.tensor([[0.0, 1.0]], device=zk.device, dtype=zk.dtype)
        else:
            k = torch.arange(N, device=zk.device, dtype=zk.dtype).view(N, 1)
            pos_forward = k / (N - 1)
            pos_backward = 1 - pos_forward
            pos_all = torch.cat([pos_forward, pos_backward], dim=1)

        pos_windows = pos_all[window_stage_indices]
        pos_windows = pos_windows.unsqueeze(0).expand(B, -1, -1, -1)
        pos_windows_flat = pos_windows.reshape(B, n_windows, 2 * self.n_nodes)

        p_expanded = p.unsqueeze(1).expand(B, n_windows, -1)
        nn_inputs = torch.cat(
            [p_expanded, Fk_windows_flat, norm_windows_flat, pos_windows_flat],
            dim=-1,
        )
        nn_inputs_flat = nn_inputs.reshape(B * n_windows, -1)

        return nn_inputs_flat, norm_Fk_stages

    def _merge_window_outputs_average(self, dz_windows: torch.Tensor) -> torch.Tensor:
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage
        B = dz_windows.shape[0]
        dz_flat = dz_windows.reshape(B, -1, n_z_stage)
        dz_sum = torch.zeros(B, N, n_z_stage, device=dz_windows.device, dtype=dz_windows.dtype)

        stage_idx = self._window_stage_indices_flat.to(device=dz_windows.device)
        stage_idx = stage_idx.view(1, -1, 1).expand(B, -1, n_z_stage)
        dz_sum.scatter_add_(dim=1, index=stage_idx, src=dz_flat)

        count = self._avg_merge_counts.to(device=dz_windows.device, dtype=dz_windows.dtype)
        return dz_sum / count.view(1, N, 1)

    def _merge_window_outputs_handoff(self, dz_windows: torch.Tensor) -> torch.Tensor:
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage
        B = dz_windows.shape[0]
        dz_stages = torch.zeros(B, N, n_z_stage, device=dz_windows.device, dtype=dz_windows.dtype)

        for k in range(N):
            w = self._handoff_owner_window_idx[k]
            j = self._handoff_owner_local_idx[k]
            dz_stages[:, k, :] = dz_windows[:, w, j, :]

        return dz_stages
    
    def forward(self, zk, p, apply_jvp: bool | None = None):
        t1=time.time()
        nn_inputs_flat, norm_Fk_stages = self._map_window_inputs(zk, p)
        t2=time.time()
        print("Mapping time: "+str(t2-t1))
        B = zk.shape[0]
        n_z_stage = self.nlp.n_z_stage

        dz_windows_flat = self.model(nn_inputs_flat)
        dz_windows = dz_windows_flat.reshape(
            B,
            self.n_windows,
            self.n_nodes,
            n_z_stage,
        )
        t3 = time.time()
        print("Network evaluation time: " + str(t3 - t2))
        if self.overlap_merge_mode == "average":
            dz_stages = self._merge_window_outputs_average(dz_windows=dz_windows)
        elif self.overlap_merge_mode == "handoff":
            dz_stages = self._merge_window_outputs_handoff(dz_windows=dz_windows)
        else:
            raise NotImplementedError

        norm_Fk_clamped = torch.clamp(norm_Fk_stages, max=self.norm_Fk_max)
        dz_stages = dz_stages * norm_Fk_clamped

        dzk = self.nlp.compose_z_batch(dz_stages)
        dzk = dzk * self.gamma

        if apply_jvp is None:
            apply_jvp = self.jvp_linesearch
        if apply_jvp:
            with torch.no_grad():
                alpha_jvp = self.nlp.alpha_batch_func(zk, p, dzk.detach())
                if alpha_jvp.dim() == 1:
                    alpha_jvp = alpha_jvp.unsqueeze(1)
                alpha_jvp = alpha_jvp.clamp(min=1e-2, max=1.0)
            dzk = dzk * alpha_jvp
        t3 = time.time()
        print("Clamping time: " + str(t3 - t2))
        return dzk

    @torch.no_grad()
    def _forward_fast_single(self, z, p):
        self._ensure_single_eval_ready()
        fk = self._eval_fk_single(z, p)

        if self._single_kkt_backend == "casadi_single":
            fk_t = torch.from_numpy(fk).to(device=self._single_nn_device, dtype=torch.float32)
            p_t = torch.from_numpy(p).to(device=self._single_nn_device, dtype=torch.float32)
        elif self._single_kkt_backend == "cusadi_batch":
            fk_t = fk.to(device=self._single_nn_device, dtype=torch.float32)
            p_t = p.to(device=self._single_nn_device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

        n_z_stage = self.nlp.n_z_stage
        fk_stages = self.nlp.decompose_Fk(fk_t)
        norm_fk_stages = torch.linalg.vector_norm(fk_stages, ord=2, dim=-1, keepdim=True) + self.offset
        norm_fk_stages_log = torch.log(norm_fk_stages)
        fk_stages_normalized = fk_stages / norm_fk_stages

        idx = self._single_window_stage_indices
        fk_windows = fk_stages_normalized[idx, :]
        norm_windows = norm_fk_stages_log[idx, :]

        fk_windows_flat = fk_windows.reshape(self.n_windows, self.n_nodes * n_z_stage)
        norm_windows_flat = norm_windows.reshape(self.n_windows, self.n_nodes)
        p_expanded = p_t.unsqueeze(0).expand(self.n_windows, -1)

        nn_inputs = torch.cat(
            [p_expanded, fk_windows_flat, norm_windows_flat, self._single_pos_windows_flat],
            dim=-1,
        )

        dz_windows_flat = self._single_model(nn_inputs)
        dz_windows = dz_windows_flat.reshape(self.n_windows, self.n_nodes, n_z_stage)

        dz_flat = dz_windows.reshape(-1, n_z_stage)
        dz_sum = torch.zeros(
            self.nlp.N, n_z_stage, device=self._single_nn_device, dtype=torch.float32
        )
        stage_idx = self._single_window_stage_indices_flat.view(-1, 1).expand(-1, n_z_stage)
        dz_sum.scatter_add_(dim=0, index=stage_idx, src=dz_flat)
        dz_stages = dz_sum / self._single_avg_merge_counts.view(self.nlp.N, 1)

        norm_fk_clamped = torch.clamp(norm_fk_stages, max=self.norm_Fk_max)
        dz_stages = dz_stages * norm_fk_clamped

        dz = self.nlp.compose_z(dz_stages)
        dz = dz * float(self.gamma)

        if self._single_kkt_backend == "casadi_single":
            return dz.detach().cpu().numpy().astype(np.float64, copy=False)

        return dz.to(device=self._single_kkt_device, dtype=self.nlp.dtype)
