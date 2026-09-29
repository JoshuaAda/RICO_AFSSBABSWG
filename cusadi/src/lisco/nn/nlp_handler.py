"""
NLP Handler for Solver Learning (LISCO)

Wraps an NMPC object and provides CasADi functions for learning-based solvers.
All NLP evaluations (KKT, Fischer-Burmeister, merit function) are computed in CasADi,
with batching via ca.Function.map(). PyTorch is only used for tensor conversion.

Standard Form NLP:
    min_w f(w, p)
    s.t. h(w, p) = 0
         g(w, p) <= 0

Variable convention:
- w: Primal decision variables (n_w,)
- p: Parameters (n_p,)
- nu: Lagrange multipliers for equality constraints h (n_h,)
- lam: Lagrange multipliers for inequality constraints g (n_g,), lam >= 0
- z: Stacked primal-dual vector [w, nu, lam]
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class NLPConfig:
    """Configuration for NLPHandler. Defaults match NLPHandler.__init__."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64
    eps: float = 1e-8
    sigma: float = 1e-2
    n_threads: int = 16
    distributed_KKT_functions: bool = False


# Functions
batched_dot = torch.func.vmap(torch.dot, chunk_size=None)


def casadi_to_torch(
    ca_func: ca.Function,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    output_squeeze: bool = False,
) -> callable:
    """Wrap a CasADi function as a PyTorch function.

    Handles numpy/torch conversion for seamless PyTorch integration.
    Works with both single and mapped (batched) CasADi functions.

    Args:
        ca_func: CasADi function to wrap. Can be a mapped function.
        device: PyTorch device for output tensors ('cpu' or 'cuda').
        dtype: PyTorch dtype for output tensors.
        output_squeeze: If True, flatten scalar outputs to 1D tensor.

    Returns:
        A callable that accepts PyTorch tensors (batch_size, dim) and
        returns PyTorch tensor outputs with batch dimension first.

    Example:
        >>> f_ca = ca.Function("f", [x, p], [x**2 + p])
        >>> f_mapped = f_ca.map(1024, "thread", 8)
        >>> f_torch = casadi_to_torch(f_mapped, device="cuda")
        >>> result = f_torch(x_batch, p_batch)
    """

    def torch_func(*args: torch.Tensor) -> torch.Tensor:
        # Convert torch tensors to transposed numpy arrays (CasADi expects columns)
        np_args = [arg.detach().cpu().numpy().T for arg in args]

        # Evaluate CasADi function
        result_np = np.array(ca_func(*np_args))

        # Handle output shape based on dimensions
        if result_np.ndim == 2:
            # Vector output: (n, batch_size) -> (batch_size, n)
            result_np = result_np.T
        elif result_np.ndim == 3:
            # Matrix output: (m, n, batch_size) -> (batch_size, m, n)
            result_np = np.transpose(result_np, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported output shape: {result_np.shape}")

        # Handle scalar outputs
        if output_squeeze and result_np.shape[-1] == 1:
            result_np = result_np.squeeze(-1)

        return torch.tensor(result_np, dtype=dtype, device=device)

    return torch_func


# NLP Handler Class
class NLPHandler:
    """NLP Handler for solver learning.

    Wraps an NMPC object and provides CasADi functions for learning-based
    optimization solvers. All NLP evaluations are done in CasADi with
    batch support via ca.Function.map().

    Attributes:
        n_w: Number of primal decision variables.
        n_h: Number of equality constraints.
        n_g: Number of inequality constraints.
        n_z: Total primal-dual dimension (n_w + n_g + n_h).
        n_p: Number of parameters.
        device: PyTorch device ('cpu' or 'cuda').
        dtype: PyTorch dtype (torch.float32 or torch.float64).
        eps: Smoothing parameter for Fischer-Burmeister.
    """

    def __init__(
        self,
        nmpc,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-8,
        sigma: float = 1e-2,
        n_threads: int = 16,
        distributed_KKT_functions: bool = False,
    ) -> None:
        """Initialize NLPHandler from an NMPC object.

        Args:
            nmpc: NMPC instance providing nlp dict and bounds.
            device: PyTorch device ('cpu' or 'cuda').
            dtype: PyTorch dtype for tensors.
            eps: Smoothing parameter for Fischer-Burmeister.
            sigma: Regularization parameter for dual variables in constraints.
            n_threads: Number of threads for parallel CasADi evaluation.
            distributed_KKT_functions: If True, use stage-wise distributed KKT-family
                functions (requires ``nmpc.distributed_nlp``).
        """
        self.nmpc = nmpc
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.sigma = sigma
        self.batch_size = None
        self.n_threads = n_threads
        self.distributed_KKT_functions = bool(distributed_KKT_functions)
        self.distributed_nlp = getattr(self.nmpc, "distributed_nlp", None)

        # Extract standard NLP dict
        self.nlp = nmpc.nlp

        # Extract dimensions
        self._extract_dimensions()

        # Build CasADi symbolic expressions and functions
        self._build_casadi_functions()
        if self.distributed_KKT_functions:
            if self.distributed_nlp is None:
                raise ValueError(
                    "distributed_KKT_functions=True requires nmpc.distributed_nlp."
                )
            self._build_distributed_casadi_functions()

    # -------------------------------------------------------------------------
    # Dimension Extraction
    # -------------------------------------------------------------------------

    def _extract_dimensions(self) -> None:
        """Extract dimensions from nlp dict."""
        nlp = self.nlp

        self.n_w = nlp["n_w"]
        self.n_h = nlp["n_h"]
        self.n_g = nlp["n_g"]
        self.n_p = nlp["n_p"]
        self.n_z = nlp["n_z"]

        self.N = self.nmpc.N
        self.nx = self.nmpc.nx
        self.nu = self.nmpc.nu+self.nmpc.nz

        self.n_wx = (self.N + 1) * self.nx
        self.n_wu = self.N * (self.nu)

        # Per-stage dimensions for decomposition
        self.n_w_stage = self.nx + self.nu  # [u_k, x_{k+1}] per stage
        self.n_h_stage = self.nx  # dynamics constraint per stage
        # Note: n_g_stage computed in _build_stage_indices()

        # Build stage indices for decomposition
        #self._build_stage_indices()

        # Optional distributed-stage metadata from NMPC model
        if self.distributed_nlp is not None:
            dn = self.distributed_nlp
            self.n_w_stage = int(dn.get("n_w_stage", self.n_w_stage))
            self.n_h_stage = int(dn.get("n_h_stage", self.n_h_stage))
            self.n_g_stage = int(dn.get("n_g_stage", self.n_g_stage))
            self.n_ub_per_stage = int(dn.get("n_g_upper_stage", self.n_ub_per_stage))
            self.n_lb_per_stage = int(dn.get("n_g_lower_stage", self.n_lb_per_stage))
            self.n_g_upper = self.N * self.n_ub_per_stage
            self.n_g_lower = self.N * self.n_lb_per_stage
            self.n_z_stage = self.n_w_stage + self.n_h_stage + self.n_g_stage
            self.n_p_stage_start = int(dn.get("n_p_stage_start", self.nx))
            self.n_p_stage_i = int(dn.get("n_p_stage_i", self.nx))
            self.n_p_stage_end = int(dn.get("n_p_stage_end", self.nx))

            if self.n_w != self.N * self.n_w_stage:
                raise ValueError(
                    f"Inconsistent n_w ({self.n_w}) vs N*n_w_stage ({self.N * self.n_w_stage})."
                )
            if self.n_h != self.N * self.n_h_stage:
                raise ValueError(
                    f"Inconsistent n_h ({self.n_h}) vs N*n_h_stage ({self.N * self.n_h_stage})."
                )
            if self.n_g != self.N * self.n_g_stage:
                raise ValueError(
                    f"Inconsistent n_g ({self.n_g}) vs N*n_g_stage ({self.N * self.n_g_stage})."
                )

    def _build_stage_indices(self) -> None:
        """Build index mappings for decomposing z into per-stage z_k vectors.
        
        Assumes uniform bound structure per stage (same bounds on u_k and x_{k+1} for all k).
        This allows decomposition via simple reshapes.
        
        g is structured as [all_upper_bounds, all_lower_bounds] where each half
        is ordered by stage: [bounds_stage_0, bounds_stage_1, ...].
        """
        N, nx, nu, n_w = self.N, self.nx, self.nu, self.n_w
        
        # Count finite bounds per stage (using stage 0 as reference)
        x_lb, x_ub = self.nmpc.x_lb, self.nmpc.x_ub
        u_lb, u_ub = self.nmpc.u_lb, self.nmpc.u_ub
        
        self.n_ub_per_stage = int(np.isfinite(u_ub).sum() + np.isfinite(x_ub).sum())
        self.n_lb_per_stage = int(np.isfinite(u_lb).sum() + np.isfinite(x_lb).sum())
        self.n_g_upper = N * self.n_ub_per_stage
        self.n_g_lower = N * self.n_lb_per_stage
        self.n_g_stage = self.n_ub_per_stage + self.n_lb_per_stage
        self.n_z_stage = self.n_w_stage + self.n_h_stage + self.n_g_stage
        self.n_p_stage = 3 * self.nx + self.nu  # [x_k, x_{k+2}, u_{k+1}, nu_{k+1}]

        # Build constraint-to-w mapping for efficient dlamk computation
        # g_to_w_idx[i] gives the w index that constraint i depends on
        # dg_dw_sign[i] gives +1 for upper bounds, -1 for lower bounds
        self._build_constraint_mapping()

    def _build_constraint_mapping(self) -> None:
        """Build mapping from constraint indices to w indices for box constraints.
        
        For box constraints:
        - Upper bounds: g_i = w_j - ub_j <= 0  →  dg_i/dw_j = +1
        - Lower bounds: g_i = lb_j - w_j <= 0  →  dg_i/dw_j = -1
        
        This mapping enables efficient element-wise computation of dlamk.
        """
        # Build bound vectors matching w structure: [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        N = self.N
        lbw_list = [self.nmpc.u_lb]  # u_0
        ubw_list = [self.nmpc.u_ub]
        for k in range(1, N):
            lbw_list.append(self.nmpc.x_lb)
            ubw_list.append(self.nmpc.x_ub)
            lbw_list.append(self.nmpc.u_lb)
            ubw_list.append(self.nmpc.u_ub)
        lbw_list.append(self.nmpc.x_lb)  # x_N
        ubw_list.append(self.nmpc.x_ub)
        
        lbw = np.concatenate(lbw_list)
        ubw = np.concatenate(ubw_list)
        
        # Build index lists
        g_to_w_idx = []
        dg_dw_sign = []
        
        # Upper bounds first (order matches g construction in NMPC)
        for i in range(self.n_w):
            if np.isfinite(ubw[i]):
                g_to_w_idx.append(i)
                dg_dw_sign.append(1.0)  # dg/dw = +1 for upper bounds
        
        # Then lower bounds
        for i in range(self.n_w):
            if np.isfinite(lbw[i]):
                g_to_w_idx.append(i)
                dg_dw_sign.append(-1.0)  # dg/dw = -1 for lower bounds
        
        # Store as tensors for efficient GPU computation
        self.g_to_w_idx = torch.tensor(g_to_w_idx, dtype=torch.long, device=self.device)
        self.dg_dw_sign = torch.tensor(dg_dw_sign, dtype=self.dtype, device=self.device)
        
        assert len(g_to_w_idx) == self.n_g, f"Mapping size {len(g_to_w_idx)} != n_g {self.n_g}"

    # -------------------------------------------------------------------------
    # CasADi Function Building
    # -------------------------------------------------------------------------

    def _build_casadi_functions(self) -> None:
        """Build CasADi symbolic expressions and functions."""
        nlp = self.nlp

        # Extract symbolic expressions from NMPC
        w_sym = nlp["w"]
        p_sym = nlp["p"]
        f_sym = nlp["f"]
        h_sym = nlp["h"]
        g_sym = nlp["g"]

        # Primal-dual symbolic variables
        lam_sym = ca.SX.sym("lam", self.n_g)
        nu_sym = ca.SX.sym("nu", self.n_h)
        z_sym = ca.vertcat(w_sym, nu_sym, lam_sym)        

        # =====================================================================
        # Lagrangian, Gradient, and KKT Conditions
            
        # Lagrangian: L = f + nu^T h + lam^T g
        L_sym = f_sym + ca.dot(nu_sym, h_sym) + ca.dot(lam_sym, g_sym)
        
        # Gradient of Lagrangian w.r.t. w
        grad_L_sym = ca.gradient(L_sym, w_sym)

        # Hessian of Lagrangian w.r.t. w
        hess_L_sym = ca.hessian(L_sym, w_sym)[0]

        # KKT Conditions        
        g_pos = ca.fmax(0, g_sym)
        lam_neg = ca.fmin(0, lam_sym)
        comp_slack = lam_sym * g_sym
        KKT_sym = ca.vertcat(grad_L_sym, h_sym, g_pos, lam_neg, comp_slack)

        # Regularized/penalized Fischer-Burmeister
        # Standard FB: lam - g - sqrt(lam² + g² + eps²)
        fb_standard = lam_sym - g_sym - ca.sqrt(lam_sym**2 + g_sym**2 + self.eps**2)
        # Penalization: max(0, lam) * max(0, -g)
        fb_penalty = ca.fmax(0, lam_sym) * ca.fmax(0, -g_sym)
        # Combined: 80% standard + 20% penalty
        fb_sym = 0.8 * fb_standard + 0.2 * fb_penalty
        # fb_sym = lam_sym - g_sym - ca.sqrt((lam_sym + g_sym)**2 + self.eps**2)
        Fk_sym = ca.vertcat(grad_L_sym, h_sym, fb_sym)
        # Fk_sym = ca.vertcat(grad_L_sym, h_sym, ca.fmin(lam_sym, -g_sym)) # Only works with w_sigmoid and lam_softplus options

        # Jacobian of Fk w.r.t. z: DFk = ∂Fk/∂z, shape (n_z, n_z)
        DFk_sym = ca.jacobian(Fk_sym, z_sym)

        # Merit Function Tk = 0.5 * ||Fk||²
        Tk_sym = 0.5 * ca.dot(Fk_sym, Fk_sym)
        # Tk_sym = 0.5 * ca.dot(KKT_sym, KKT_sym)

        grad_Tk_z_sym = ca.gradient(Tk_sym, z_sym)

        # dfb_dw_sym = ca.jacobian(fb_sym, w_sym)
        # dfb_dlam_sym = ca.jacobian(fb_sym, lam_sym)

        dz_sym = ca.SX.sym("dz", self.n_z)
        jvp_sym = ca.jtimes(Fk_sym, z_sym, dz_sym, False)  # J(z,p) @ dz
        num = ca.dot(Fk_sym, jvp_sym)                      # scalar
        den = ca.dot(jvp_sym, jvp_sym) + 1e-16             # scalar

        alpha_sym = -num / den                             # scalar (can be negative)
        self.alpha_func_ca = ca.Function("alpha", [z_sym, p_sym, dz_sym], [alpha_sym])
    
        # Create CasADi functions
        self.f_func_ca = ca.Function("f_fn", [w_sym, p_sym], [f_sym])
        self.h_func_ca = ca.Function("h_fn", [w_sym, p_sym], [h_sym])
        self.g_func_ca = ca.Function("g_fn", [w_sym, p_sym], [g_sym])
        self.fb_func_ca = ca.Function("fb_fn", [z_sym, p_sym], [fb_sym])
        self.comp_slack_func_ca = ca.Function("comp_slack_fn", [z_sym, p_sym], [comp_slack])
        self.L_func_ca = ca.Function("L_fn", [w_sym, nu_sym, lam_sym, p_sym], [L_sym])
        self.grad_L_func_ca = ca.Function("grad_L_fn", [z_sym, p_sym], [grad_L_sym])
        self.hess_L_func_ca = ca.Function("hess_L_fn", [z_sym, p_sym], [hess_L_sym])
        self.KKT_func_ca = ca.Function("KKT_fn", [z_sym, p_sym], [KKT_sym])
        self.Fk_func_ca = ca.Function("Fk_fn", [z_sym, p_sym], [Fk_sym])
        self.DFk_func_ca = ca.Function("DFk_fn", [z_sym, p_sym], [DFk_sym])
        self.Tk_nodiff_func_ca = ca.Function("Tk_nodiff_fn", [z_sym, p_sym], [Tk_sym])
        self.grad_Tk_z_func_ca = ca.Function("grad_Tk_z_fn", [z_sym, p_sym], [grad_Tk_z_sym])
        # self.dfb_dw_func_ca = ca.Function("dfb_dw_fn", [z_sym, p_sym], [dfb_dw_sym])
        # self.dfb_dlam_func_ca = ca.Function("dfb_dlam_fn", [z_sym, p_sym], [dfb_dlam_sym])

        # =====================================================================
        # Fischer-Burmeister Residual (Fk) - Regularized & Penalized
        # fb = 0.8 * (lam - g - sqrt(lam² + g² + eps)) + 0.2 * max(0, lam) * max(0, -g)
        # =====================================================================

        # # Regularized constraints
        # nu_bar_sym = ca.SX.sym("nu_bar", self.n_h)
        # lam_bar_sym = ca.SX.sym("lam_bar", self.n_g)
        # h_reg_sym = h_sym - self.sigma * (nu_sym - nu_bar_sym)
        # g_reg_sym = g_sym - self.sigma * (lam_sym - lam_bar_sym)

        # fb_reg_sym = 0.8 * (lam_sym - g_reg_sym - ca.sqrt(lam_sym**2 + g_reg_sym**2 + self.eps**2)) + 0.2 * ca.fmax(0, lam_sym) * ca.fmax(0, -g_reg_sym)
        # Fk_reg_sym = ca.vertcat(grad_L_sym, h_reg_sym, fb_reg_sym)

        # Tk_reg_sym = 0.5 * ca.dot(Fk_reg_sym, Fk_reg_sym)
        # grad_Tk_reg_z_sym = ca.gradient(Tk_reg_sym, z_sym)
        # self.Tk_nodiff_reg_func_ca = ca.Function("Tk_reg_nodiff_fn", [z_sym, p_sym, nu_bar_sym, lam_bar_sym], [Tk_reg_sym])
        # self.grad_Tk_z_reg_func_ca = ca.Function("grad_Tk_reg_z_fn", [z_sym, p_sym, nu_bar_sym, lam_bar_sym], [grad_Tk_reg_z_sym])

        # =====================================================================
        # Augmented Lagrangian Loss Vk (Primal-Only, Well-Conditioned)
        # Vk = f(w) + nu_bar^T h(w) + lam_bar^T g(w) + (rho/2)||h||² + (rho/2)||[g]_+||²
        # 
        # nu_bar, lam_bar are detached dual variables (no gradient flow)
        # This avoids Hessian conditioning issues in backprop - only constraint
        # Jacobians are needed for gradients.
        # =====================================================================
        
        # rho = 1.0  # Fixed penalty parameter
        
        # g_plus_sym = ca.fmax(0, g_sym)  # [g]_+ = max(0, g)
        # Vk_sym = (f_sym 
        #           + ca.dot(nu_bar_sym, h_sym) 
        #           + ca.dot(lam_bar_sym, g_sym)
        #           + 0.5 * rho * ca.dot(h_sym, h_sym)
        #           + 0.5 * rho * ca.dot(g_plus_sym, g_plus_sym))
        
        # # Gradient of Vk w.r.t. w only (primal variables)
        # grad_Vk_w_sym = ca.gradient(Vk_sym, w_sym)
        
        # self.Vk_nodiff_func_ca = ca.Function(
        #     "Vk_nodiff_fn", 
        #     [w_sym, p_sym, nu_bar_sym, lam_bar_sym], 
        #     [Vk_sym]
        # )
        # self.grad_Vk_w_func_ca = ca.Function(
        #     "grad_Vk_w_fn", 
        #     [w_sym, p_sym, nu_bar_sym, lam_bar_sym], 
        #     [grad_Vk_w_sym]
        # )

    # -------------------------------------------------------------------------
    # Distributed CasADi Function Building
    # -------------------------------------------------------------------------

    def _build_distributed_casadi_functions(self) -> None:
        """Build distributed stage-local CasADi functions from nmpc.distributed_nlp."""
        dn = self.distributed_nlp
        required_keys = (
            "w_stage",
            "p_stage_start",
            "p_stage_i",
            "p_stage_end",
            "f_start",
            "f_i",
            "f_end",
            "h_start",
            "h_i",
            "h_end",
            "g_i",
        )
        missing = [k for k in required_keys if k not in dn]
        if missing:
            raise ValueError(f"distributed_nlp missing keys: {missing}")

        if self.n_p_stage_start != self.nx or self.n_p_stage_i != self.nx or self.n_p_stage_end != self.nx:
            raise NotImplementedError(
                "Current distributed handler expects stage parameters p_stage_* to be x_k (size nx)."
            )

        w_stage_base = dn["w_stage"]
        p_stage_start_base = dn["p_stage_start"]
        p_stage_i_base = dn["p_stage_i"]
        p_stage_end_base = dn["p_stage_end"]
        f_start_base = dn["f_start"]
        f_i_base = dn["f_i"]
        f_end_base = dn["f_end"]
        h_start_base = dn["h_start"]
        h_i_base = dn["h_i"]
        h_end_base = dn["h_end"]
        g_i_base = dn["g_i"]

        self.n_kkt_stage = self.n_w_stage + self.n_h_stage + 3 * self.n_g_stage
        self._dist_stage_roles = ("start", "mid", "end")
        self._dist_output_meta: dict[tuple[str, str], dict] = {}
        self._dist_role_p_dim_model = {
            "start": self.n_p_stage_start,
            "mid": self.n_p_stage_i,
            "end": self.n_p_stage_end,
        }
        self._dist_role_has_nu_next = {"start": True, "mid": True, "end": False}

        role_data = {
            "start": (f_start_base, h_start_base, p_stage_start_base),
            "mid": (f_i_base, h_i_base, p_stage_i_base),
            "end": (f_end_base, h_end_base, p_stage_end_base),
        }

        for role in self._dist_stage_roles:
            f_base, h_base, p_base = role_data[role]
            p_dim_model = self._dist_role_p_dim_model[role]
            has_nu_next = self._dist_role_has_nu_next[role]
            p_dim_total = p_dim_model + (self.n_h_stage if has_nu_next else 0)

            z_stage_sym = ca.SX.sym(f"z_stage_{role}", self.n_z_stage)
            p_stage_sym = ca.SX.sym(f"p_stage_{role}", p_dim_total)

            w_stage_sym = z_stage_sym[: self.n_w_stage]
            x_next_sym = w_stage_sym[self.nu : self.nu + self.nx]
            nu_stage_sym = z_stage_sym[self.n_w_stage : self.n_w_stage + self.n_h_stage]
            lam_stage_sym = z_stage_sym[self.n_w_stage + self.n_h_stage :]

            p_model_sym = p_stage_sym[:p_dim_model]
            nu_next_sym = (
                p_stage_sym[p_dim_model : p_dim_model + self.n_h_stage]
                if has_nu_next
                else ca.SX.zeros(0, 1)
            )

            f_stage_sym = ca.substitute(f_base, w_stage_base, w_stage_sym)
            h_stage_sym = ca.substitute(
                h_base,
                ca.vertcat(w_stage_base, p_base),
                ca.vertcat(w_stage_sym, p_model_sym),
            )
            g_stage_sym = ca.substitute(g_i_base, w_stage_base, w_stage_sym)

            L_stage_sym = (
                f_stage_sym
                + ca.dot(nu_stage_sym, h_stage_sym)
                + ca.dot(lam_stage_sym, g_stage_sym)
            )
            if has_nu_next:
                L_stage_sym = L_stage_sym - ca.dot(nu_next_sym, x_next_sym)

            grad_L_stage_sym = ca.gradient(L_stage_sym, w_stage_sym)

            g_pos_sym = ca.fmax(0, g_stage_sym)
            lam_neg_sym = ca.fmin(0, lam_stage_sym)
            comp_slack_stage_sym = lam_stage_sym * g_stage_sym
            KKT_stage_sym = ca.vertcat(
                grad_L_stage_sym, h_stage_sym, g_pos_sym, lam_neg_sym, comp_slack_stage_sym
            )

            fb_standard = (
                lam_stage_sym
                - g_stage_sym
                - ca.sqrt(lam_stage_sym**2 + g_stage_sym**2 + self.eps**2)
            )
            fb_penalty = ca.fmax(0, lam_stage_sym) * ca.fmax(0, -g_stage_sym)
            fb_stage_sym = 0.8 * fb_standard + 0.2 * fb_penalty
            Fk_stage_sym = ca.vertcat(grad_L_stage_sym, h_stage_sym, fb_stage_sym)
            Tk_stage_sym = 0.5 * ca.dot(Fk_stage_sym, Fk_stage_sym)

            y_stage_sym = ca.vertcat(z_stage_sym, p_stage_sym)
            grad_Tk_stage_sym = ca.gradient(Tk_stage_sym, y_stage_sym)

            kkt_fn = ca.Function(
                f"KKT_stage_{role}_fn", [z_stage_sym, p_stage_sym], [KKT_stage_sym]
            )
            fk_fn = ca.Function(
                f"Fk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [Fk_stage_sym]
            )
            tk_fn = ca.Function(
                f"Tk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [Tk_stage_sym]
            )
            grad_fn = ca.Function(
                f"grad_Tk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [grad_Tk_stage_sym]
            )

            setattr(self, f"KKT_stage_{role}_func_ca", kkt_fn)
            setattr(self, f"Fk_stage_{role}_func_ca", fk_fn)
            setattr(self, f"Tk_stage_{role}_func_ca", tk_fn)
            setattr(self, f"grad_Tk_stage_{role}_func_ca", grad_fn)

            for fam_name, fn, dense_size in (
                ("KKT", kkt_fn, self.n_kkt_stage),
                ("Fk", fk_fn, self.n_z_stage),
                ("Tk", tk_fn, 1),
                ("grad_Tk", grad_fn, self.n_z_stage + p_dim_total),
            ):
                rows = np.array(fn.sparsity_out(0).get_triplet()[0], dtype=np.int64)
                self._dist_output_meta[(fam_name, role)] = {
                    "dense_size": int(dense_size),
                    "row_idx": torch.tensor(rows, dtype=torch.long, device=self.device),
                }

    def _distributed_stage_counts(self, batch_size: int) -> dict[str, int]:
        if self.N == 1:
            return {"start": 0, "mid": 0, "end": batch_size}
        return {
            "start": batch_size,
            "mid": batch_size * max(self.N - 2, 0),
            "end": batch_size,
        }

    def _normalize_distributed_output(
        self, raw: torch.Tensor, fam_name: str, role: str, n_rows: int
    ) -> torch.Tensor:
        raw_t = raw.to(dtype=self.dtype, device=self.device)
        if raw_t.ndim == 1:
            raw_t = raw_t.unsqueeze(-1)
        elif raw_t.ndim != 2:
            raise ValueError(
                f"Unsupported distributed tensor rank for {fam_name}/{role}: "
                f"shape={tuple(raw_t.shape)}"
            )

        if raw_t.shape[0] != n_rows:
            if raw_t.shape[1] == n_rows:
                raw_t = raw_t.T
            else:
                raise ValueError(
                    f"Distributed output row mismatch for {fam_name}/{role}: "
                    f"got {tuple(raw_t.shape)}, expected first dim {n_rows}"
                )

        meta = self._dist_output_meta[(fam_name, role)]
        dense_size = int(meta["dense_size"])
        row_idx = meta["row_idx"].to(device=raw_t.device)

        if raw_t.shape[1] == dense_size:
            return raw_t
        if raw_t.shape[1] == row_idx.numel():
            dense = torch.zeros(n_rows, dense_size, dtype=raw_t.dtype, device=raw_t.device)
            dense[:, row_idx] = raw_t
            return dense

        raise ValueError(
            f"Unsupported distributed output shape for {fam_name}/{role}: "
            f"got {tuple(raw_t.shape)}, dense_size={dense_size}, nnz={row_idx.numel()}"
        )

    def _distributed_prepare_stage_context(
        self, z: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = z.shape[0]
        if self.batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size}. "
                "Call setup_for_batch with the desired batch size first."
            )
        if p.shape[1] != self.nx:
            raise ValueError(
                f"Distributed stage context currently expects p dimension nx={self.nx}, "
                f"got p.shape[1]={p.shape[1]}."
            )

        z_stages = self.decompose_z_batch(z)
        x_next = z_stages[:, :, self.nu : self.nu + self.nx]
        nu_dual = z_stages[:, :, self.n_w_stage : self.n_w_stage + self.n_h_stage]

        x_prev = torch.cat([p.unsqueeze(1), x_next[:, :-1, :]], dim=1)
        nu_next = torch.cat(
            [
                nu_dual[:, 1:, :],
                torch.zeros(
                    batch_size, 1, self.n_h_stage, dtype=self.dtype, device=self.device
                ),
            ],
            dim=1,
        )
        return z_stages, x_prev, nu_next, batch_size

    def _distributed_eval_role(
        self, fam_name: str, role: str, z_in: torch.Tensor, p_in: torch.Tensor
    ) -> torch.Tensor:
        eval_fn = getattr(self, f"_{fam_name}_stage_{role}_eval")
        if eval_fn is None:
            raise RuntimeError(f"Distributed eval function not initialized for {fam_name}/{role}")
        raw = eval_fn(z_in, p_in)
        return self._normalize_distributed_output(raw, fam_name, role, z_in.shape[0])

    def _distributed_eval_stage_family(
        self, z: torch.Tensor, p: torch.Tensor, fam_name: str, out_dim: int
    ) -> torch.Tensor:
        z_stages, x_prev, nu_next, batch_size = self._distributed_prepare_stage_context(z, p)
        out_stages = torch.zeros(batch_size, self.N, out_dim, dtype=self.dtype, device=self.device)

        if self.N == 1:
            z_end = z_stages[:, 0, :]
            p_end = x_prev[:, 0, :]
            out_stages[:, 0, :] = self._distributed_eval_role(fam_name, "end", z_end, p_end)
            return out_stages

        z_start = z_stages[:, 0, :]
        p_start = torch.cat([x_prev[:, 0, :], nu_next[:, 0, :]], dim=-1)
        out_stages[:, 0, :] = self._distributed_eval_role(fam_name, "start", z_start, p_start)

        if self.N > 2:
            z_mid = z_stages[:, 1:-1, :].reshape(batch_size * (self.N - 2), self.n_z_stage)
            p_mid = torch.cat([x_prev[:, 1:-1, :], nu_next[:, 1:-1, :]], dim=-1).reshape(
                batch_size * (self.N - 2), self.n_p_stage_i + self.n_h_stage
            )
            mid_out = self._distributed_eval_role(fam_name, "mid", z_mid, p_mid)
            out_stages[:, 1:-1, :] = mid_out.reshape(batch_size, self.N - 2, out_dim)

        z_end = z_stages[:, -1, :]
        p_end = x_prev[:, -1, :]
        out_stages[:, -1, :] = self._distributed_eval_role(fam_name, "end", z_end, p_end)
        return out_stages

    def _compose_stage_g_block_batch(self, block_stages: torch.Tensor) -> torch.Tensor:
        """Compose one stage-wise g-like block to global [upper_all, lower_all] ordering."""
        upper = block_stages[:, :, : self.n_ub_per_stage].reshape(block_stages.shape[0], -1)
        lower = block_stages[:, :, self.n_ub_per_stage :].reshape(block_stages.shape[0], -1)
        return torch.cat([upper, lower], dim=-1)

    def _assemble_distributed_kkt(self, KKT_stage_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Assemble full KKT vector from stage-wise KKT outputs."""
        KKT_stages = KKT_stage_flat.reshape(batch_size, self.N, self.n_kkt_stage)

        i0 = self.n_w_stage
        i1 = i0 + self.n_h_stage
        i2 = i1 + self.n_g_stage
        i3 = i2 + self.n_g_stage

        grad_stages = KKT_stages[:, :, :i0]
        h_stages = KKT_stages[:, :, i0:i1]
        g_pos_stages = KKT_stages[:, :, i1:i2]
        lam_neg_stages = KKT_stages[:, :, i2:i3]
        comp_stages = KKT_stages[:, :, i3:]

        grad_global = grad_stages.reshape(batch_size, -1)
        h_global = h_stages.reshape(batch_size, -1)
        g_pos_global = self._compose_stage_g_block_batch(g_pos_stages)
        lam_neg_global = self._compose_stage_g_block_batch(lam_neg_stages)
        comp_global = self._compose_stage_g_block_batch(comp_stages)

        return torch.cat(
            [grad_global, h_global, g_pos_global, lam_neg_global, comp_global], dim=-1
        )

    def _assemble_distributed_grad_Tk(
        self, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Assemble global grad_Tk from stage-local gradients."""
        z_stages, x_prev, nu_next, batch_size = self._distributed_prepare_stage_context(z, p)
        grad_z_stages = torch.zeros(batch_size, self.N, self.n_z_stage, dtype=self.dtype, device=self.device)

        x_slice = slice(self.nu, self.nu + self.nx)
        nu_slice = slice(self.n_w_stage, self.n_w_stage + self.n_h_stage)

        if self.N == 1:
            grad_end = self._distributed_eval_role(
                "grad_Tk", "end", z_stages[:, 0, :], x_prev[:, 0, :]
            )
            grad_z_stages[:, 0, :] += grad_end[:, : self.n_z_stage]
            return self.compose_z_batch(grad_z_stages)

        grad_start = self._distributed_eval_role(
            "grad_Tk",
            "start",
            z_stages[:, 0, :],
            torch.cat([x_prev[:, 0, :], nu_next[:, 0, :]], dim=-1),
        )
        grad_z_stages[:, 0, :] += grad_start[:, : self.n_z_stage]
        grad_nu_next_start = grad_start[:, self.n_z_stage + self.n_p_stage_start :]
        grad_z_stages[:, 1, nu_slice] += grad_nu_next_start

        if self.N > 2:
            z_mid = z_stages[:, 1:-1, :].reshape(batch_size * (self.N - 2), self.n_z_stage)
            p_mid = torch.cat([x_prev[:, 1:-1, :], nu_next[:, 1:-1, :]], dim=-1).reshape(
                batch_size * (self.N - 2), self.n_p_stage_i + self.n_h_stage
            )
            grad_mid = self._distributed_eval_role("grad_Tk", "mid", z_mid, p_mid).reshape(
                batch_size, self.N - 2, self.n_z_stage + self.n_p_stage_i + self.n_h_stage
            )
            grad_z_stages[:, 1:-1, :] += grad_mid[:, :, : self.n_z_stage]

            grad_x_prev_mid = grad_mid[:, :, self.n_z_stage : self.n_z_stage + self.n_p_stage_i]
            grad_nu_next_mid = grad_mid[:, :, self.n_z_stage + self.n_p_stage_i :]
            grad_z_stages[:, 0 : self.N - 2, x_slice] += grad_x_prev_mid
            grad_z_stages[:, 2:self.N, nu_slice] += grad_nu_next_mid

        grad_end = self._distributed_eval_role(
            "grad_Tk", "end", z_stages[:, -1, :], x_prev[:, -1, :]
        )
        grad_z_stages[:, -1, :] += grad_end[:, : self.n_z_stage]
        grad_x_prev_end = grad_end[:, self.n_z_stage : self.n_z_stage + self.n_p_stage_end]
        grad_z_stages[:, self.N - 2, x_slice] += grad_x_prev_end

        return self.compose_z_batch(grad_z_stages)

    def _bind_distributed_interfaces(self) -> None:
        """Bind distributed batch/single-sample callable interfaces."""

        def Fk_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Fk_stages = self._distributed_eval_stage_family(
                z, p, fam_name="Fk", out_dim=self.n_z_stage
            )
            return self.compose_Fk_batch(Fk_stages)

        def KKT_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            KKT_stages = self._distributed_eval_stage_family(
                z, p, fam_name="KKT", out_dim=self.n_kkt_stage
            )
            batch_size = z.shape[0]
            KKT_stage_flat = KKT_stages.reshape(batch_size * self.N, self.n_kkt_stage)
            return self._assemble_distributed_kkt(KKT_stage_flat, batch_size)

        def Tk_nodiff_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Tk_stages = self._distributed_eval_stage_family(
                z, p, fam_name="Tk", out_dim=1
            )
            return Tk_stages[:, :, 0].sum(dim=1)

        def grad_Tk_z_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return self._assemble_distributed_grad_Tk(z, p)

        def Tk_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Tk_nodiff_batch = Tk_nodiff_batch_func(z, p)
            grad_Tk_batch = grad_Tk_z_batch_func(z, p)
            return Tk_nodiff_batch + batched_dot(grad_Tk_batch, z - z.detach())

        def Fk_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return Fk_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        def KKT_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return KKT_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        def Tk_nodiff_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return Tk_nodiff_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        def grad_Tk_z_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return grad_Tk_z_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        def Tk_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return Tk_batch_func(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        self.Fk_batch_func = Fk_batch_func
        self.KKT_batch_func = KKT_batch_func
        self.Tk_batch_nodiff_func = Tk_nodiff_batch_func
        self.grad_Tk_z_batch_func = grad_Tk_z_batch_func
        self.Tk_batch_func = Tk_batch_func

        self.Fk_func = Fk_func
        self.KKT_func = KKT_func
        self.Tk_nodiff_func = Tk_nodiff_func
        self.grad_Tk_z_func = grad_Tk_z_func
        self.Tk_func = Tk_func

    def _setup_distributed_batch_functions(self, batch_size: int, n_threads: int = 1) -> None:
        """Setup distributed stage-mapped CasADi functions for batch evaluation."""
        self.batch_size = batch_size
        self.n_threads = n_threads

        stage_counts = self._distributed_stage_counts(batch_size)
        for role, n_instances in stage_counts.items():
            for fam in ("KKT", "Fk", "Tk", "grad_Tk"):
                setattr(self, f"_{fam}_stage_{role}_eval", None)
                setattr(self, f"_{fam}_stage_{role}_batch_ca", None)
            if n_instances <= 0:
                continue

            kkt_map = getattr(self, f"KKT_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            fk_map = getattr(self, f"Fk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            tk_map = getattr(self, f"Tk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            grad_map = getattr(self, f"grad_Tk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )

            setattr(self, f"_KKT_stage_{role}_batch_ca", kkt_map)
            setattr(self, f"_Fk_stage_{role}_batch_ca", fk_map)
            setattr(self, f"_Tk_stage_{role}_batch_ca", tk_map)
            setattr(self, f"_grad_Tk_stage_{role}_batch_ca", grad_map)

            setattr(
                self,
                f"_KKT_stage_{role}_eval",
                casadi_to_torch(kkt_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_Fk_stage_{role}_eval",
                casadi_to_torch(fk_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_Tk_stage_{role}_eval",
                casadi_to_torch(tk_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_grad_Tk_stage_{role}_eval",
                casadi_to_torch(grad_map, device=self.device, dtype=self.dtype),
            )

        self._bind_distributed_interfaces()

    def _setup_distributed_cusadi_functions(
        self, batch_size: int = 1024, force_rebuild: bool = False
    ) -> None:
        """Override distributed stage evaluators with GPU CusADi callables."""
        import sys

        if sys.platform != "linux":
            return
        if not str(self.device).startswith("cuda"):
            return

        print("Setting up distributed Cusadi functions for batch evaluation...")
        from apply_cusadi import apply_cusadi

        stage_counts = self._distributed_stage_counts(batch_size)
        for role, n_instances in stage_counts.items():
            if n_instances <= 0:
                continue

            setattr(
                self,
                f"_KKT_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"KKT_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"KKT_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_Fk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"Fk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"Fk_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_Tk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"Tk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"Tk_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_grad_Tk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"grad_Tk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"grad_Tk_stage_{role}_batch",
                ),
            )
        self.g_batch_func = apply_cusadi(
            self.g_func_ca,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
            name="g_batch",
        )

    # -------------------------------------------------------------------------
    # Batch Function Setup
    # -------------------------------------------------------------------------

    def set_new_eps(self, new_eps: float) -> None:
        """Update FB smoothing epsilon and rebuild dependent functions."""
        new_eps = float(new_eps)
        if (not np.isfinite(new_eps)) or (new_eps <= 0.0):
            raise ValueError(f"new_eps must be a finite positive float, got {new_eps}")

        if new_eps == self.eps:
            return

        self.eps = new_eps

        self._build_casadi_functions()
        if self.distributed_KKT_functions:
            self._build_distributed_casadi_functions()

        if self.batch_size is not None:
            self.setup_for_batch(batch_size=self.batch_size, force_rebuild=False)

    def _setup_batch_functions(self, batch_size: int, n_threads: int = 1) -> None:
        """Setup batch-mapped CasADi functions for parallel evaluation."""
        self.batch_size = batch_size
        self.n_threads = n_threads

        self._KKT_batch_ca = self.KKT_func_ca.map(batch_size, "thread", n_threads)
        self._Fk_batch_ca = self.Fk_func_ca.map(batch_size, "thread", n_threads)
        self._Tk_batch_nodiff_ca = self.Tk_nodiff_func_ca.map(batch_size, "thread", n_threads)
        self._grad_Tk_z_batch_ca = self.grad_Tk_z_func_ca.map(batch_size, "thread", n_threads)

        # self._Tk_batch_nodiff_reg_ca = self.Tk_nodiff_reg_func_ca.map(batch_size, "thread", n_threads)
        # self._grad_Tk_z_batch_reg_ca = self.grad_Tk_z_reg_func_ca.map(batch_size, "thread", n_threads)

        self.f_batch_ca = self.f_func_ca.map(batch_size, "thread", n_threads)
        self.h_batch_func_ca = self.h_func_ca.map(batch_size, "thread", n_threads)
        self.g_batch_func_ca = self.g_func_ca.map(batch_size, "thread", n_threads)
        self.fb_batch_func_ca = self.fb_func_ca.map(batch_size, "thread", n_threads)
        self.comp_slack_batch_func_ca = self.comp_slack_func_ca.map(batch_size, "thread", n_threads)
        self.grad_L_batch_func_ca = self.grad_L_func_ca.map(batch_size, "thread", n_threads)

        # self.alpha_batch_func_ca = self.alpha_func_ca.map(batch_size, "thread", n_threads)

        # Augmented Lagrangian batch functions
        # self._Vk_batch_nodiff_ca = self.Vk_nodiff_func_ca.map(batch_size, "thread", n_threads)
        # self._grad_Vk_w_batch_ca = self.grad_Vk_w_func_ca.map(batch_size, "thread", n_threads)

    # -------------------------------------------------------------------------
    # PyTorch Interface: Batch Functions
    # -------------------------------------------------------------------------

    def _setup_torch_functions(self) -> None:
        self.f_func = casadi_to_torch(self.f_func_ca, device=self.device, dtype=self.dtype)
        self.f_batch_func = casadi_to_torch(self.f_batch_ca, device=self.device, dtype=self.dtype)
        self.h_func = casadi_to_torch(self.h_func_ca, device=self.device, dtype=self.dtype)
        self.g_func = casadi_to_torch(self.g_func_ca, device=self.device, dtype=self.dtype)
        self.L_func = casadi_to_torch(self.L_func_ca, device=self.device, dtype=self.dtype)
        self.grad_L_func = casadi_to_torch(self.grad_L_func_ca, device=self.device, dtype=self.dtype)
        self.hess_L_func = casadi_to_torch(self.hess_L_func_ca, device=self.device, dtype=self.dtype)
        self.KKT_func = casadi_to_torch(self.KKT_func_ca, device=self.device, dtype=self.dtype)
        self.Fk_func = casadi_to_torch(self.Fk_func_ca, device=self.device, dtype=self.dtype)
        self.Tk_nodiff_func = casadi_to_torch(self.Tk_nodiff_func_ca, device=self.device, dtype=self.dtype)
        self.grad_Tk_z_func = casadi_to_torch(self.grad_Tk_z_func_ca, device=self.device, dtype=self.dtype)
        self.KKT_batch_func = casadi_to_torch(self._KKT_batch_ca, device=self.device, dtype=self.dtype)
        self.Fk_batch_func = casadi_to_torch(self._Fk_batch_ca, device=self.device, dtype=self.dtype)
        
        self.grad_L_batch_func = casadi_to_torch(self.grad_L_batch_func_ca, device=self.device, dtype=self.dtype)
        self.h_batch_func = casadi_to_torch(self.h_batch_func_ca, device=self.device, dtype=self.dtype)
        self.g_batch_func = casadi_to_torch(self.g_batch_func_ca, device=self.device, dtype=self.dtype)
        self.fb_batch_func = casadi_to_torch(self.fb_batch_func_ca, device=self.device, dtype=self.dtype)
        self.comp_slack_batch_func = casadi_to_torch(self.comp_slack_batch_func_ca, device=self.device, dtype=self.dtype)

        self.Tk_batch_nodiff_func = casadi_to_torch(self._Tk_batch_nodiff_ca, device=self.device, dtype=self.dtype)
        self.grad_Tk_z_batch_func = casadi_to_torch(self._grad_Tk_z_batch_ca, device=self.device, dtype=self.dtype)

        # self.Tk_batch_nodiff_reg_func = casadi_to_torch(self._Tk_batch_nodiff_reg_ca, device=self.device, dtype=self.dtype)
        # self.grad_Tk_z_batch_reg_func = casadi_to_torch(self._grad_Tk_z_batch_reg_ca, device=self.device, dtype=self.dtype)

        self.DFk_func = casadi_to_torch(self.DFk_func_ca, device=self.device, dtype=self.dtype)

        self.alpha_func = casadi_to_torch(self.alpha_func_ca, device=self.device, dtype=self.dtype)
        # self.alpha_batch_func = casadi_to_torch(self.alpha_batch_func_ca, device=self.device, dtype=self.dtype)
        
        # self.dfb_dw_func = casadi_to_torch(self.dfb_dw_func_ca, device=self.device, dtype=self.dtype)
        # self.dfb_dlam_func = casadi_to_torch(self.dfb_dlam_func_ca, device=self.device, dtype=self.dtype)

    def Tk_func(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Evaluate merit function Tk with first-order correction."""
        with torch.no_grad():     
            z_np = z.cpu().numpy().T
            p_np = p.cpu().numpy().T
            Tk_nodiff_pt = torch.from_numpy(np.array(self.Tk_nodiff_func_ca(z_np, p_np))).to(dtype=self.dtype, device=self.device).squeeze()
            grad_Tk_z_pt = torch.from_numpy(np.array(self.grad_Tk_z_func_ca(z_np, p_np)).T).to(dtype=self.dtype, device=self.device).squeeze()
        z_diff = z - z.detach()
        # Fk_pt = torch.from_numpy(np.array(self.Fk_func_ca(z_np, p_np)).T).to(dtype=self.dtype, device=self.device).squeeze()
        Tk_pt = Tk_nodiff_pt + torch.dot(grad_Tk_z_pt, z_diff) # + torch.dot(self.sigma*Fk_pt,z_diff)
        return Tk_pt

    def Tk_batch_func(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        z_np = z.detach().cpu().numpy().T
        p_np = p.detach().cpu().numpy().T
        Tk_nodiff_pt = torch.from_numpy(np.array(self._Tk_batch_nodiff_ca(z_np, p_np))).to(dtype=self.dtype, device=self.device).squeeze()
        grad_Tk_z_pt = torch.from_numpy(np.array(self._grad_Tk_z_batch_ca(z_np, p_np)).T).to(dtype=self.dtype, device=self.device).squeeze()
        z_diff = z - z.detach()
        # Fk_batch_pt = torch.from_numpy(np.array(self._Fk_batch_ca(z_np, p_np))).to(dtype=self.dtype, device=self.device).squeeze()
        Tk_batch_pt = Tk_nodiff_pt + batched_dot(grad_Tk_z_pt, z_diff) # + batched_dot(self.sigma*Fk_batch_pt,z_diff)
        return Tk_batch_pt
    
    # def Tk_func(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    #     """Evaluate merit function Tk with first-order correction."""
    #     with torch.no_grad():     
    #         z_np = z.cpu().numpy().T
    #         p_np = p.cpu().numpy().T
    #         w,nu,lam = self.extract_primal_dual(z)
    #         nu_np = nu.cpu().numpy().T
    #         lam_np = lam.cpu().numpy().T
    #         Tk_nodiff_reg_pt = torch.from_numpy(np.array(self.Tk_nodiff_reg_func_ca(z_np, p_np, nu_np, lam_np))).to(dtype=self.dtype, device=self.device).squeeze()
    #         grad_Tk_z_reg_pt = torch.from_numpy(np.array(self.grad_Tk_z_reg_func_ca(z_np, p_np, nu_np, lam_np)).T).to(dtype=self.dtype, device=self.device).squeeze()
    #     z_diff = z - z.detach()
    #     Tk_pt = Tk_nodiff_reg_pt + torch.dot(grad_Tk_z_reg_pt, z_diff)
    #     return Tk_pt

    # def Tk_batch_func(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    #     z_np = z.detach().cpu().numpy().T
    #     p_np = p.detach().cpu().numpy().T
    #     w,nu,lam = self.extract_primal_dual_batch(z)
    #     nu_np = nu.detach().cpu().numpy().T
    #     lam_np = lam.detach().cpu().numpy().T
    #     Tk_nodiff_reg_pt = torch.from_numpy(np.array(self._Tk_batch_nodiff_reg_ca(z_np, p_np, nu_np, lam_np))).to(dtype=self.dtype, device=self.device).squeeze()
    #     grad_Tk_z_reg_pt = torch.from_numpy(np.array(self._grad_Tk_z_batch_reg_ca(z_np, p_np, nu_np, lam_np)).T).to(dtype=self.dtype, device=self.device).squeeze()
    #     z_diff = z - z.detach()
    #     Tk_batch_pt = Tk_nodiff_reg_pt + batched_dot(grad_Tk_z_reg_pt, z_diff)
    #     return Tk_batch_pt

    # def Vk_batch_func(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    #     """Evaluate Augmented Lagrangian loss Vk with first-order correction.
        
    #     Vk = f(w) + nu_bar^T h(w) + lam_bar^T g(w) + (rho/2)||h||² + (rho/2)||[g]_+||²
        
    #     The dual variables (nu, lam) are DETACHED - no gradient flows through them.
    #     This provides a well-conditioned loss that only requires constraint Jacobians
    #     for backpropagation, avoiding Hessian conditioning issues.
        
    #     First-order correction for differentiability w.r.t. w only:
    #     Vk(w) ≈ Vk(w_detached) + ∇_w Vk^T (w - w_detached)
        
    #     Args:
    #         z: Primal-dual variables [w, nu, lam], shape (batch_size, n_z)
    #         p: Parameters, shape (batch_size, n_p)
            
    #     Returns:
    #         Vk: Augmented Lagrangian loss, shape (batch_size,)
    #     """
    #     w, nu, lam = self.extract_primal_dual_batch(z)
        
    #     # Detach duals - treated as fixed for gradient computation
    #     nu_bar = nu.detach()
    #     lam_bar = lam.detach()
        
    #     # Numpy arrays for CasADi
    #     w_np = w.detach().cpu().numpy().T
    #     p_np = p.detach().cpu().numpy().T
    #     nu_bar_np = nu_bar.cpu().numpy().T
    #     lam_bar_np = lam_bar.cpu().numpy().T
        
    #     # Evaluate Vk and gradient at detached point
    #     with torch.no_grad():
    #         Vk_nodiff = torch.from_numpy(
    #             np.array(self._Vk_batch_nodiff_ca(w_np, p_np, nu_bar_np, lam_bar_np))
    #         ).to(dtype=self.dtype, device=self.device).squeeze()
            
    #         grad_Vk_w = torch.from_numpy(
    #             np.array(self._grad_Vk_w_batch_ca(w_np, p_np, nu_bar_np, lam_bar_np)).T
    #         ).to(dtype=self.dtype, device=self.device)
        
    #     # First-order correction: gradient only w.r.t. w (primal variables)
    #     w_diff = w - w.detach()
    #     Vk_batch = Vk_nodiff + batched_dot(grad_Vk_w, w_diff)
        
    #     return Vk_batch

    # def Vk_dual_batch_func(
    #     self, 
    #     z_old: torch.Tensor, 
    #     z_new: torch.Tensor, 
    #     p: torch.Tensor,
    #     beta: float = 1.0,
    # ) -> torch.Tensor:
    #     """Evaluate combined Augmented Lagrangian loss with KKT-based dual regularization.
        
    #     Total loss = Vk(w_new) + beta * L_dual
        
    #     where:
    #     - Vk(w_new) = f(w_new) + nu_bar^T h(w_new) + lam_bar^T g(w_new) + penalties
    #     - L_dual = ||nu_new - nu_target||² + ||lam_new ⊙ g(w_new)||² + ||relu(-lam_new)||²
    #     - nu_target = nu_old + rho * h(w_new)  (ALM update for equalities)
        
    #     The lambda loss uses complementarity (lam * g = 0) instead of ALM update,
    #     which correctly drives lam → 0 when g < 0 regardless of lam_old.
        
    #     This provides gradient signal for BOTH primal and dual predictions:
    #     - Primal (w): from Vk gradient (well-conditioned, no Hessian)
    #     - Dual (nu): from ALM update matching
    #     - Dual (lam): from complementarity + dual feasibility
        
    #     Args:
    #         z_old: Previous primal-dual variables [w_old, nu_old, lam_old], shape (B, n_z)
    #         z_new: New primal-dual variables [w_new, nu_new, lam_new], shape (B, n_z)
    #         p: Parameters, shape (B, n_p)
    #         beta: Weight for dual loss term (default 1.0)
            
    #     Returns:
    #         Combined loss, shape (B,)
    #     """
    #     rho = 1.0  # Fixed penalty parameter (same as in Vk)
        
    #     # Extract components
    #     w_old, nu_old, lam_old = self.extract_primal_dual_batch(z_old)
    #     w_new, nu_new, lam_new = self.extract_primal_dual_batch(z_new)
        
    #     # Detach old duals for Vk computation (no gradient flow through them)
    #     nu_bar = nu_old.detach()
    #     lam_bar = lam_old.detach()
        
    #     # =====================================================================
    #     # Primal loss: Vk(w_new) with gradient only w.r.t. w_new
    #     # =====================================================================
    #     w_np = w_new.detach().cpu().numpy().T
    #     p_np = p.detach().cpu().numpy().T
    #     nu_bar_np = nu_bar.cpu().numpy().T
    #     lam_bar_np = lam_bar.cpu().numpy().T
        
    #     with torch.no_grad():
    #         Vk_nodiff = torch.from_numpy(
    #             np.array(self._Vk_batch_nodiff_ca(w_np, p_np, nu_bar_np, lam_bar_np))
    #         ).to(dtype=self.dtype, device=self.device).squeeze()
            
    #         grad_Vk_w = torch.from_numpy(
    #             np.array(self._grad_Vk_w_batch_ca(w_np, p_np, nu_bar_np, lam_bar_np)).T
    #         ).to(dtype=self.dtype, device=self.device)
        
    #     w_diff = w_new - w_new.detach()
    #     Vk_primal = Vk_nodiff + batched_dot(grad_Vk_w, w_diff)
        
    #     # =====================================================================
    #     # Dual loss for nu: ALM update (works well for equalities)
    #     # nu_target = nu_old + rho * h(w_new)
    #     # =====================================================================
        
    #     # Evaluate h(w_new) and g(w_new) - detached, no gradient through constraints
    #     with torch.no_grad():
    #         h_new = self.h_batch_func(w_new.detach(), p)  # (B, n_h)
    #         g_new = self.g_batch_func(w_new.detach(), p)  # (B, n_g)
        
    #     # ALM dual update target for nu (detached)
    #     nu_target = nu_old.detach() + rho * h_new
    #     dual_loss_nu = torch.sum((nu_new - nu_target) ** 2, dim=1)
        
    #     # =====================================================================
    #     # Dual loss for lam: Complementarity + Dual Feasibility
    #     # - Complementarity: lam * g = 0 (drives lam → 0 when g < 0)
    #     # - Dual feasibility: lam >= 0
    #     # =====================================================================
        
    #     # Complementarity loss: ||lam_new ⊙ g(w_new)||²
    #     # This correctly pushes lam → 0 when g < 0, regardless of lam_old
    #     comp_loss = torch.sum((lam_new * g_new) ** 2, dim=1)
        
    #     # Dual feasibility loss: ||relu(-lam_new)||² (penalizes lam < 0)
    #     dual_feas_loss = torch.sum(torch.relu(-lam_new) ** 2, dim=1)
        
    #     # Combined dual loss
    #     dual_loss = dual_loss_nu + comp_loss + dual_feas_loss
        
    #     # Combined loss
    #     total_loss = Vk_primal + beta * dual_loss
        
    #     return total_loss

    # def compute_dlamk_fb_batch(self, z_batch: torch.Tensor, p_batch: torch.Tensor, dw_batch: torch.Tensor) -> torch.Tensor:
    #     """Compute dual step dlam for box constraints using Fischer-Burmeister.
        
    #     Efficient element-wise implementation exploiting box constraint structure.
    #     For box constraints, each constraint depends on exactly one w variable:
    #     - Upper bounds: g_i = w_j - ub_j  →  dg/dw = +1
    #     - Lower bounds: g_i = lb_j - w_j  →  dg/dw = -1
        
    #     The update formula is (from Newton on fb = 0):
    #         ∂fb/∂lam · dlam + ∂fb/∂g · dg/dw · dw = -fb
    #         dlam_i = -(fb_i + dfb_dg_i * dg_dw_i * dw_i) / (dfb_dlam_i + sigma)
        
    #     Args:
    #         z_batch: Primal-dual variables, shape (B, n_z)
    #         p_batch: Parameters, shape (B, n_p)
    #         dw_batch: Primal step, shape (B, n_w)
            
    #     Returns:
    #         dlamk: Dual step, shape (B, n_g)
    #     """
    #     # Extract w, lam from z
    #     w, nu, lam = self.extract_primal_dual_batch(z_batch)
        
    #     # Get g values: use CasADi function
    #     g_batch = self.g_batch_func(w, p_batch)  # (B, n_g)
        
    #     # Get fb values: use CasADi function
    #     fb_batch = self.fb_batch_func(z_batch, p_batch)  # (B, n_g)
        
    #     # 1. r_i = sqrt(g_i^2 + lam_i^2 + eps^2)
    #     r = torch.sqrt(g_batch**2 + lam**2 + self.eps**2)  # (B, n_g)
        
    #     # 2. dfb_dg_i = -1 - g_i / r_i
    #     dfb_dg = -1.0 - g_batch / r  # (B, n_g)

    #     dinv = 1.0
        
    #     # 6. Get dw_i for each constraint using precomputed mapping
    #     # g_to_w_idx[i] gives the w index that constraint i depends on
    #     dw_i = dw_batch[:, self.g_to_w_idx]  # (B, n_g)
        
    #     # 7. dg_dw is +1 for upper bounds, -1 for lower bounds
    #     dg_dw = self.dg_dw_sign  # (n_g,) broadcast to (B, n_g)
        
    #     # 8. Newton step: ∂fb/∂lam · dlam + ∂fb/∂g · dg/dw · dw = -fb
    #     #    Solving for dlam: dlam = -(fb + dfb_dg * dg_dw * dw_i) / dfb_dlam
    #     dlamk = - dinv * (dfb_dg * dg_dw * dw_i + fb_batch)  # (B, n_g)
        
    #     return dlamk
    
    def setup_for_batch(self, batch_size: int, force_rebuild: bool = False) -> None:
        """Single entry point for (re-)configuring batch evaluation functions.

        Sets up CPU-mapped CasADi batch functions and their PyTorch wrappers,
        then overrides the hot-path functions with GPU CusADi callables on Linux.
        On Windows the CusADi step is a no-op.

        Args:
            batch_size: Number of parallel instances for batch evaluation.
            force_rebuild: If True, force CUDA codegen even if cached artifacts exist.
        """
        if self.distributed_KKT_functions:
            self._setup_batch_functions(batch_size=batch_size, n_threads=self.n_threads)
            self._setup_torch_functions()
            self._setup_distributed_batch_functions(
                batch_size=batch_size, n_threads=self.n_threads
            )
            self._setup_distributed_cusadi_functions(
                batch_size=batch_size, force_rebuild=force_rebuild
            )
            return

        self._setup_batch_functions(batch_size=batch_size, n_threads=self.n_threads)
        self._setup_torch_functions()
        self._setup_cusadi_functions(batch_size=batch_size, force_rebuild=force_rebuild)

    def _setup_cusadi_functions(self, batch_size: int = 1024, force_rebuild: bool = False) -> None:
        """Override hot-path batch functions with GPU CusADi callables (Linux + CUDA only).

        Skipped when ``self.device`` is ``"cpu"`` or when not running on Linux.

        Args:
            batch_size: Number of parallel instances for batch evaluation.
            force_rebuild: If True, force CUDA codegen even if cached artifacts exist.
        """
        import sys
        if sys.platform != "linux":
            return
        if not str(self.device).startswith("cuda"):
            return

        print("Setting up Cusadi functions for batch evaluation...")
        from apply_cusadi import apply_cusadi

        self.Fk_batch_func = apply_cusadi(self.Fk_func_ca, batch_size=batch_size, force_rebuild=force_rebuild)
        self.Tk_nodiff_batch_func = apply_cusadi(self.Tk_nodiff_func_ca, batch_size=batch_size, force_rebuild=force_rebuild)
        self.grad_Tk_z_batch_func = apply_cusadi(self.grad_Tk_z_func_ca, batch_size=batch_size, force_rebuild=force_rebuild)
        self.g_batch_func = apply_cusadi(self.g_func_ca, batch_size=batch_size, force_rebuild=force_rebuild, name="g_batch")
        # self.alpha_batch_func = apply_cusadi(self.alpha_func_ca, batch_size=batch_size, force_rebuild=force_rebuild)

        def Tk_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            # w, nu, lam = self.extract_primal_dual_batch(z)
            Tk_nodiff_batch = self.Tk_nodiff_batch_func(z, p).to(dtype=self.dtype, device=self.device).squeeze()
            grad_Tk_z_batch = self.grad_Tk_z_batch_func(z, p).to(dtype=self.dtype, device=self.device).squeeze()
            # Fk_batch = self.Fk_batch_func(z,p).to(dtype=self.dtype, device=self.device).squeeze()
            Tk_batch = Tk_nodiff_batch + batched_dot(grad_Tk_z_batch, z - z.detach()) # + batched_dot(self.sigma*Fk_batch, z - z.detach())
            return Tk_batch

        self.Tk_batch_func = Tk_batch_func

    # -------------------------------------------------------------------------
    # Primal-Dual Extraction and Stacking
    # -------------------------------------------------------------------------

    def extract_primal_dual(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract primal and dual variables from z = [w, nu, lam]."""
        w = z[: self.n_w]
        nu = z[self.n_w : self.n_w + self.n_h]
        lam = z[self.n_w + self.n_h :]
        return w, nu, lam

    def stack_primal_dual(
        self, w: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Stack primal and dual variables into z = [w, nu, lam]."""
        return torch.hstack([w, nu, lam])

    def extract_primal_dual_batch(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract primal and dual variables from batched z."""
        w = z[:, : self.n_w]
        nu = z[:, self.n_w : self.n_w + self.n_h]
        lam = z[:, self.n_w + self.n_h :]
        return w, nu, lam

    def stack_primal_dual_batch(
        self, w: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Stack primal and dual variables into batched z."""
        return torch.hstack([w, nu, lam])

    @torch.no_grad()
    def project_z_box(self, z: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """Project z = [w, nu, lam] onto w in [0,1] and lam >= 0 (nu unbounded)."""
        squeeze = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True

        w, nu, lam = self.extract_primal_dual_batch(z)
        w = w.clamp(min=float(eps), max=1.0 - float(eps))
        
        lam = lam.clamp_min(float(eps))
        # if hasattr(self, "g_to_w_idx"):
        #     # Handle both 1D and 2D (batched) lam
        #     lam_size = lam.shape[-1] if lam.dim() > 1 else lam.shape[0]
        #     if self.g_to_w_idx.numel() == lam_size:
        #         w_bnd = (w <= float(eps)) | (w >= 1.0 - float(eps))
                
        #         if w.dim() == 1:
        #             # Single sample case
        #             lam = lam.masked_fill(w_bnd[self.g_to_w_idx], 0.0)
        #         else:
        #             # Batch case
        #             idx = self.g_to_w_idx.to(device=w.device).view(1, -1).expand(w.shape[0], -1)
        #             lam = lam.masked_fill(w_bnd.gather(dim=1, index=idx), 0.0)
        
        # if lam.numel() > 0:
        #     lam = lam.clamp_min(float(eps))
        #     if hasattr(self, "g_to_w_idx") and self.g_to_w_idx.numel() == lam.shape[1]:
        #         w_bnd = (w <= float(eps)) | (w >= 1.0 - float(eps))
        #         idx = self.g_to_w_idx.to(device=w.device).view(1, -1).expand(w.shape[0], -1)
        #         lam = lam.masked_fill(w_bnd.gather(dim=1, index=idx), 0.0)
        # z_proj = self.stack_primal_dual_batch(w, nu, lam)

        z_proj = self.stack_primal_dual_batch(w, nu, lam)
        if squeeze:
            return z_proj.squeeze(0)
        return z_proj

    # -------------------------------------------------------------------------
    # Time-Stage Decomposition
    # -------------------------------------------------------------------------

    def decompose_z(self, z: torch.Tensor) -> torch.Tensor:
        """Decompose primal-dual vector z into per-stage vectors z_k.
        
        Args:
            z: Primal-dual vector of shape (n_z,)
            
        Returns:
            z_stages: Tensor of shape (N, n_z_stage)
        """
        w, nu, lam = self.extract_primal_dual(z)
        N, nx, nu_dim = self.N, self.nx, self.nu
        
        w_stages = w.view(N, nu_dim + nx)
        nu_stages = nu.view(N, nx)
        lam_upper, lam_lower = lam.split([self.n_g_upper, self.n_g_lower])
        lam_stages = torch.cat([lam_upper.view(N, -1), lam_lower.view(N, -1)], dim=-1)
        
        return torch.cat([w_stages, nu_stages, lam_stages], dim=-1)

    def decompose_z_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Decompose batched primal-dual vectors z into per-stage vectors.
        
        Args:
            z: Batched primal-dual vectors of shape (B, n_z)
            
        Returns:
            z_stages: Tensor of shape (B, N, n_z_stage)
        """
        B = z.shape[0]
        w, nu, lam = self.extract_primal_dual_batch(z)
        N, nx, nu_dim = self.N, self.nx, self.nu
        
        w_stages = w.view(B, N, nu_dim + nx)
        nu_stages = nu.view(B, N, nx)
        lam_upper, lam_lower = lam.split([self.n_g_upper, self.n_g_lower], dim=-1)
        lam_stages = torch.cat([lam_upper.view(B, N, -1), lam_lower.view(B, N, -1)], dim=-1)
        
        return torch.cat([w_stages, nu_stages, lam_stages], dim=-1)

    def compose_z(self, z_stages: torch.Tensor) -> torch.Tensor:
        """Compose per-stage vectors z_k back into full primal-dual vector z.
        
        Args:
            z_stages: Tensor of shape (N, n_z_stage)
            
        Returns:
            z: Primal-dual vector of shape (n_z,)
        """
        N, nx, nu_dim = self.N, self.nx, self.nu
        n_w_s, n_h_s = self.n_w_stage, self.n_h_stage
        
        w = z_stages[:, :n_w_s].reshape(-1)
        nu = z_stages[:, n_w_s:n_w_s + n_h_s].reshape(-1)
        lam_stages = z_stages[:, n_w_s + n_h_s:]
        lam_upper = lam_stages[:, :self.n_ub_per_stage].reshape(-1)
        lam_lower = lam_stages[:, self.n_ub_per_stage:].reshape(-1)
        lam = torch.cat([lam_upper, lam_lower])
        
        return self.stack_primal_dual(w, nu, lam)

    def compose_z_batch(self, z_stages: torch.Tensor) -> torch.Tensor:
        """Compose batched per-stage vectors z_k back into full primal-dual vectors.
        
        Args:
            z_stages: Tensor of shape (B, N, n_z_stage)
            
        Returns:
            z: Batched primal-dual vectors of shape (B, n_z)
        """
        B = z_stages.shape[0]
        N, nx, nu_dim = self.N, self.nx, self.nu
        n_w_s, n_h_s = self.n_w_stage, self.n_h_stage
        
        w = z_stages[:, :, :n_w_s].reshape(B, -1)
        nu = z_stages[:, :, n_w_s:n_w_s + n_h_s].reshape(B, -1)
        lam_stages = z_stages[:, :, n_w_s + n_h_s:]
        lam_upper = lam_stages[:, :, :self.n_ub_per_stage].reshape(B, -1)
        lam_lower = lam_stages[:, :, self.n_ub_per_stage:].reshape(B, -1)
        lam = torch.cat([lam_upper, lam_lower], dim=-1)
        
        return self.stack_primal_dual_batch(w, nu, lam)

    # Aliases for Fk decomposition (same structure as z)
    decompose_Fk = decompose_z
    decompose_Fk_batch = decompose_z_batch
    compose_Fk = compose_z
    compose_Fk_batch = compose_z_batch

    # -------------------------------------------------------------------------
    # Pseudo-Parameters for Decomposed Solver
    # -------------------------------------------------------------------------

    def build_pseudo_params_batch(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Build per-stage pseudo-parameters p_k = [x_k, x_{k+2}, u_{k+1}, nu_{k+1}].
        
        Boundary: k=0 uses p; k=N-1 uses x_{k+1}, u_k, zeros for nu.
        """
        B, N, nx, nu = z.shape[0], self.N, self.nx, self.nu
        z_s = self.decompose_z_batch(z)  # (B, N, n_z_stage)
        
        u = z_s[:, :, :nu]
        x_next = z_s[:, :, nu:nu + nx]
        nu_dual = z_s[:, :, self.n_w_stage:self.n_w_stage + nx]
        
        x_k = torch.cat([p.unsqueeze(1), x_next[:, :-1]], dim=1)
        x_k2 = torch.cat([x_next[:, 1:], x_next[:, -1:]], dim=1)
        u_k1 = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        nu_k1 = torch.cat([nu_dual[:, 1:], torch.zeros(B, 1, nx, dtype=self.dtype, device=self.device)], dim=1)
        
        return torch.cat([x_k, x_k2, u_k1, nu_k1], dim=-1)

    def build_pseudo_params(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Single-sample version of build_pseudo_params_batch."""
        return self.build_pseudo_params_batch(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

    # -------------------------------------------------------------------------
    # State/Control Extraction and Stacking
    # -------------------------------------------------------------------------

    def extract_states_controls(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract states and controls from w.
        
        Structure: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Returns:
            X: States tensor of shape (N, nx) for x_1, ..., x_N
            U: Controls tensor of shape (N, nu) for u_0, ..., u_{N-1}
        """
        N, nx, nu = self.N, self.nx, self.nu
        X_list, U_list = [], []
        idx = 0
        
        # u_0
        U_list.append(w[idx : idx + nu])
        idx += nu
        
        # x_1, u_1, ..., x_{N-1}, u_{N-1}
        for k in range(1, N):
            X_list.append(w[idx : idx + nx])
            idx += nx
            U_list.append(w[idx : idx + nu])
            idx += nu
        
        # x_N
        X_list.append(w[idx : idx + nx])
        
        return torch.stack(X_list), torch.stack(U_list)

    def stack_states_controls(
        self, X: torch.Tensor, U: torch.Tensor
    ) -> torch.Tensor:
        """Stack states and controls into w.
        
        Structure: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Args:
            X: States tensor of shape (N, nx) for x_1, ..., x_N
            U: Controls tensor of shape (N, nu) for u_0, ..., u_{N-1}
        """
        N, nx, nu = self.N, self.nx, self.nu
        if X.dim() == 1:
            X = X.view(N, nx)
        if U.dim() == 1:
            U = U.view(N, nu)
        
        w_parts = [U[0]]  # u_0
        for k in range(1, N):
            w_parts.append(X[k - 1])  # x_k
            w_parts.append(U[k])      # u_k
        w_parts.append(X[N - 1])  # x_N
        return torch.cat(w_parts)

    # -------------------------------------------------------------------------
    # Batch Generation for Training
    # -------------------------------------------------------------------------

    def batch_gen_x0(self, n_batch: int, offset: float = 0.0,p: float=0.05) -> torch.Tensor:
        """Generate random initial states in [0, 1] (scaled).
        
        Args:
            n_batch: Number of samples to generate.
            offset: Fraction to extend sampling range beyond [0, 1].
                    With offset=0.1, samples from [-0.1, 1.1].
        """
        lb = -4.0-offset
        ub = 4.0 + offset
        x0 = lb + (ub - lb) * torch.rand(n_batch, self.nx, device=self.device, dtype=self.dtype)
        array=np.array([0,1])
        p=np.array([1-p,p])
        if torch.is_tensor(n_batch):
            n=np.array(n_batch.cpu())
        else:
            n=n_batch
        z_lb=torch.tensor(np.random.choice(array,p=p,size=(n,2*self.N))).to(self.device)#torch.randint(low=0,high=2,size=(n_batch, 2*self.N),device=self.device, dtype=self.dtype)#torch.zeros(n_batch, 2*self.N, device=self.device, dtype=self.dtype)
        z_ub=torch.tensor(np.random.choice(array,p=1-p,size=(n,2*self.N))).to(self.device)
        #z_ub=torch.max(z_lb,torch.randint(low=0,high=2,size=(n_batch, 2*self.N),device=self.device, dtype=self.dtype))#torch.ones(n_batch, 2*self.N, device=self.device, dtype=self.dtype)

        #z_bounds=torch.randint(low=0,high=2,size=(n_batch, 4*self.N),device=self.device, dtype=self.dtype)
        p=torch.concatenate((x0,z_lb,z_ub),dim=1)
        return p

    def batch_gen_p(self, n_batch: int, offset: float = 0.0,p: float=0.05) -> torch.Tensor:
        """Generate random parameter samples (p = x0 for NMPC)."""
        return self.batch_gen_x0(n_batch, offset=offset,p=p)

    def batch_gen_w(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random primal variable samples in [0, 1] (scaled).
        
        Structure: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Note: x_0 is a parameter, not a decision variable.
        
        Args:
            n_batch: Number of samples to generate.
            offset: Fraction to extend sampling range beyond [0, 1].
        """
        lb = -offset
        ub = 1.0 + offset

        # U: N control inputs, X: N states (x_1, ..., x_N)
        U = lb + (ub - lb) * torch.rand(n_batch, self.N, self.nu, device=self.device, dtype=self.dtype)
        X = lb + (ub - lb) * torch.rand(n_batch, self.N, self.nx, device=self.device, dtype=self.dtype)

        # Stack into w = [u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N]
        w_list = [U[:, 0, :]]  # u_0
        for k in range(1, self.N):
            w_list.append(X[:, k - 1, :])  # x_k
            w_list.append(U[:, k, :])      # u_k
        w_list.append(X[:, self.N - 1, :])  # x_N
        w = torch.cat(w_list, dim=1)
        return w

    def batch_gen_z(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random primal-dual variable samples."""
        w = self.batch_gen_w(n_batch, offset=offset)
        nu = torch.rand(n_batch, self.n_h, device=self.device, dtype=self.dtype)
        # lam = torch.rand(n_batch, self.n_g, device=self.device, dtype=self.dtype)
        lam = torch.zeros(n_batch, self.n_g, device=self.device, dtype=self.dtype)
        # w = 0.5*torch.ones(n_batch, self.n_w, device=self.device, dtype=self.dtype)
        # nu = torch.zeros(n_batch, self.n_h, device=self.device, dtype=self.dtype)
        # lam = torch.zeros(n_batch, self.n_g, device=self.device, dtype=self.dtype)
        return self.stack_primal_dual_batch(w, nu, lam)


    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_dimensions(self) -> dict[str, int]:
        """Get all dimension information."""
        return {
            "n_w": self.n_w, "n_h": self.n_h, "n_g": self.n_g,
            "n_z": self.n_z, "n_p": self.n_p, "N": self.N,
            "nx": self.nx, "nu": self.nu,
            # Per-stage dimensions
            "n_w_stage": self.n_w_stage,
            "n_h_stage": self.n_h_stage,
            "n_g_stage": self.n_g_stage,
            "n_z_stage": self.n_z_stage,
        }

    def export_hparams(self) -> dict:
        """Export hyperparameters for logging."""
        return {
            "N": self.N, "n_w": self.n_w, "n_h": self.n_h, "n_g": self.n_g,
            "n_z": self.n_z, "eps": self.eps, "sigma": self.sigma,
            "device": self.device, "dtype": str(self.dtype), 
            "batch_size": self.batch_size,
            "distributed_KKT_functions": self.distributed_KKT_functions,
            # Per-stage dimensions
            "n_w_stage": self.n_w_stage,
            "n_h_stage": self.n_h_stage,
            "n_g_stage": self.n_g_stage,
            "n_z_stage": self.n_z_stage,
        }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NLPHandler Test")
    print("=" * 60)

    from models.cstr_nmpc import CSTRNMPC, NMPCParams
    from models.cstr_cascade import CSTRParams

    model_params = CSTRParams(nr=2, dt=1.0)
    nmpc_params = NMPCParams(N=50)
    nmpc = CSTRNMPC(model_params, nmpc_params)

    n_batch = 1

    print("\nCreating NLPHandler...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = NLPHandler(nmpc, device=device, dtype=torch.float64, n_threads=1,distributed_KKT_functions=False)
    nlp.setup_for_batch(batch_size=n_batch)

    # KKT Functions
    x0 = model_params.get_default_x0()
    x0_scaled = nmpc._scale_x(x0)  # Scale x0 to [0, 1]
    z_opt, info = nmpc.solve(x0_scaled)
    w_opt, nu_opt, lam_opt = nlp.extract_primal_dual(z_opt)
    KKT_val = np.array(nlp.KKT_func_ca(z_opt, x0_scaled))
    Fk_val = nlp.Fk_func_ca(z_opt, x0_scaled)
    assert np.abs(KKT_val).max() < 1e-7, "KKT conditions not satisfied at optimum!"
    assert np.abs(Fk_val).max() < 1e-5, "Fk not close to zero at optimum!"

    dims = nlp.get_dimensions()
    print(f"\nDimensions: {dims}")
    
    # nlp.setup_batch_functions(n_batch, n_threads=1)

    p_batch = nlp.batch_gen_p(n_batch)
    z_batch = nlp.batch_gen_z(n_batch)
    print(f"\nz_batch shape: {z_batch.shape}, p_batch shape: {p_batch.shape}")

    KKT_vals = nlp.KKT_batch_func(z_batch, p_batch)
    print(f"KKT shape: {KKT_vals.shape}, norm mean={torch.norm(KKT_vals, dim=1).mean():.4e}")

    Fk_vals = nlp.Fk_batch_func(z_batch, p_batch)
    print(f"Fk shape: {Fk_vals.shape}, norm mean={torch.norm(Fk_vals, dim=1).mean():.4e}")

    Tk_vals = nlp.Tk_batch_func(z_batch, p_batch)
    print(f"Tk values: mean={Tk_vals.mean():.4e}, std={Tk_vals.std():.4e}")

    # Note: Tk_batch_func uses regularized Fk_reg, not Fk, and is normalized by n_z.
    # Check that Tk values are positive (as expected for a squared norm).
    assert (Tk_vals >= 0).all(), "Tk values should be non-negative!"
    
    # Verify relationship: Tk_nodiff = 0.5 * ||Fk||^2 / n_z (for non-regularized version)
    Tk_nodiff_vals = nlp.Tk_batch_nodiff_func(z_batch, p_batch).squeeze()
    expected_Tk_nodiff = 0.5 * batched_dot(Fk_vals, Fk_vals) #/ nlp.n_z
    assert torch.allclose(expected_Tk_nodiff, Tk_nodiff_vals), "Tk_nodiff values do not match 0.5 * ||Fk||^2 !"
    print(f"Verified: Tk_nodiff = 0.5 * ||Fk||^2 ")
    # print(f"Verified: Tk_nodiff = 0.5 * ||Fk||^2 / n_z ✓")

    print("\n" + "=" * 60)
    print("NLPHandler test complete!")
    print("=" * 60)

    # Use standard autograd instead of torch.func.grad
    z_test = z_batch[0, :].clone().requires_grad_(True)
    p_test = p_batch[0, :]

    Tk_val = nlp.Tk_func(z_test, p_test)
    Tk_val.backward()
    grad_z = z_test.grad

    # Compare with CasADi gradient (must use regularized version since Tk_func uses Tk_reg)
    w, nu, lam = nlp.extract_primal_dual(z_batch[0, :])
    z_np = z_batch[0:1, :].detach().cpu().numpy().T
    p_np = p_batch[0:1, :].cpu().numpy().T
    nu_np = nu.cpu().numpy().reshape(-1, 1)
    lam_np = lam.cpu().numpy().reshape(-1, 1)
    grad_z_ca = torch.from_numpy(np.array(nlp.grad_Tk_z_func_ca(z_np, p_np)).T).to(dtype=nlp.dtype, device=nlp.device).squeeze()
    
    print(f"Gradient shape: {grad_z.shape}")
    print(f"Gradient error: {torch.norm(grad_z - grad_z_ca):.2e}")
    print(f"Max abs error: {torch.abs(grad_z - grad_z_ca).max():.2e}")

    # Use standard autograd instead of torch.func.grad
    z_test = z_batch.clone().requires_grad_(True)
    p_test = p_batch

    Tk_vals = nlp.Tk_batch_func(z_test, p_test)
    loss = Tk_vals.sum()
    loss.backward()
    grad_z = z_test.grad

    # Compare with CasADi gradient (must use regularized version since Tk_batch_func uses Tk_reg)
    w, nu, lam = nlp.extract_primal_dual_batch(z_batch)
    z_np = z_batch.detach().cpu().numpy().T
    p_np = p_batch.cpu().numpy().T
    nu_np = nu.cpu().numpy().T
    lam_np = lam.cpu().numpy().T
    grad_z_ca = torch.from_numpy(
        np.array(nlp._grad_Tk_z_batch_ca(z_np, p_np)).T
    ).to(dtype=nlp.dtype, device=nlp.device).squeeze()

    print(f"Gradient shape: {grad_z.shape}")
    print(f"Gradient error: {torch.norm(grad_z - grad_z_ca):.2e}")
    print(f"Max abs error: {torch.abs(grad_z - grad_z_ca).max():.2e}")

    # =========================================================================
    # Test Time-Stage Decomposition
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing Time-Stage Decomposition")
    print("=" * 60)

    # Print stage dimensions
    print(f"\nStage dimensions:")
    print(f"  n_w_stage = {nlp.n_w_stage} (nu={nlp.nu} + nx={nlp.nx})")
    print(f"  n_h_stage = {nlp.n_h_stage}")
    print(f"  n_g_stage = {nlp.n_g_stage}")
    print(f"  n_z_stage = {nlp.n_z_stage}")
    print(f"  N stages  = {nlp.N}")

    # -------------------------------------------------------------------------
    # Create labeled test vector for visual verification
    # w_k → k, nu_k → k*11, lam_k → k*111
    # -------------------------------------------------------------------------
    print("\n--- Visual verification with labeled values ---")
    N = nlp.N
    w_labeled = torch.zeros(nlp.n_w, dtype=nlp.dtype, device=nlp.device)
    nu_labeled = torch.zeros(nlp.n_h, dtype=nlp.dtype, device=nlp.device)
    lam_labeled = torch.zeros(nlp.n_g, dtype=nlp.dtype, device=nlp.device)
    
    # Fill w: each stage k gets value k
    for k in range(N):
        w_labeled[k * nlp.n_w_stage : (k+1) * nlp.n_w_stage] = k
    
    # Fill nu: each stage k gets value k*11 (0, 11, 22, ...)
    for k in range(N):
        nu_labeled[k * nlp.n_h_stage : (k+1) * nlp.n_h_stage] = k * 11
    
    # Fill lam: upper bounds then lower bounds, each stage k gets value k*111
    # lam = [upper_stage_0, upper_stage_1, ..., lower_stage_0, lower_stage_1, ...]
    for k in range(N):
        # Upper bounds for stage k
        lam_labeled[k * nlp.n_ub_per_stage : (k+1) * nlp.n_ub_per_stage] = k * 111
        # Lower bounds for stage k
        lam_labeled[nlp.n_g_upper + k * nlp.n_lb_per_stage : nlp.n_g_upper + (k+1) * nlp.n_lb_per_stage] = k * 111
    
    z_labeled = nlp.stack_primal_dual(w_labeled, nu_labeled, lam_labeled)
    
    print(f"z_labeled shape: {z_labeled.shape}")
    print(f"w  (first 21): {w_labeled[:21].tolist()}")
    print(f"nu (first 15): {nu_labeled[:15].tolist()}")
    print(f"lam upper (first 21): {lam_labeled[:21].tolist()}")
    print(f"lam lower (first 21): {lam_labeled[nlp.n_g_upper:nlp.n_g_upper+21].tolist()}")
    
    # Decompose and print
    z_stages_labeled = nlp.decompose_z(z_labeled)
    print(f"\nDecomposed z_stages shape: {z_stages_labeled.shape}")
    print("Per-stage values (each row should have consistent k, k*11, k*111):")
    for k in range(min(N, 5)):  # Print first 5 stages
        z_k = z_stages_labeled[k]
        w_k = z_k[:nlp.n_w_stage]
        nu_k = z_k[nlp.n_w_stage : nlp.n_w_stage + nlp.n_h_stage]
        lam_k = z_k[nlp.n_w_stage + nlp.n_h_stage:]
        print(f"  Stage {k}: w_k={w_k[0]:.0f}, nu_k={nu_k[0]:.0f}, lam_k={lam_k[0]:.0f}")
    
    # Verify correctness
    for k in range(N):
        z_k = z_stages_labeled[k]
        assert torch.allclose(z_k[:nlp.n_w_stage], torch.full((nlp.n_w_stage,), float(k), dtype=nlp.dtype, device=nlp.device)), f"w mismatch at stage {k}"
        assert torch.allclose(z_k[nlp.n_w_stage:nlp.n_w_stage+nlp.n_h_stage], torch.full((nlp.n_h_stage,), float(k*11), dtype=nlp.dtype, device=nlp.device)), f"nu mismatch at stage {k}"
        assert torch.allclose(z_k[nlp.n_w_stage+nlp.n_h_stage:], torch.full((nlp.n_g_stage,), float(k*111), dtype=nlp.dtype, device=nlp.device)), f"lam mismatch at stage {k}"
    print("✓ All stages have correct labeled values!")

    # Test single decomposition
    z_single = z_batch[0, :]
    z_stages = nlp.decompose_z(z_single)
    print(f"\nSingle decomposition:")
    print(f"  Input z shape: {z_single.shape}")
    print(f"  Output z_stages shape: {z_stages.shape}")
    assert z_stages.shape == (nlp.N, nlp.n_z_stage), f"Expected ({nlp.N}, {nlp.n_z_stage})"

    # Test compose (inverse)
    z_recomposed = nlp.compose_z(z_stages)
    print(f"  Recomposed z shape: {z_recomposed.shape}")
    decomp_error = torch.norm(z_single - z_recomposed)
    print(f"  Decompose-Compose error: {decomp_error:.2e}")
    assert decomp_error < 1e-10, "Decompose-Compose should be identity!"

    # Test batch decomposition
    z_stages_batch = nlp.decompose_z_batch(z_batch)
    print(f"\nBatch decomposition:")
    print(f"  Input z_batch shape: {z_batch.shape}")
    print(f"  Output z_stages_batch shape: {z_stages_batch.shape}")
    assert z_stages_batch.shape == (n_batch, nlp.N, nlp.n_z_stage)

    # Test batch compose (inverse)
    z_recomposed_batch = nlp.compose_z_batch(z_stages_batch)
    print(f"  Recomposed z_batch shape: {z_recomposed_batch.shape}")
    batch_decomp_error = torch.norm(z_batch - z_recomposed_batch)
    print(f"  Batch Decompose-Compose error: {batch_decomp_error:.2e}")
    assert batch_decomp_error < 1e-10, "Batch Decompose-Compose should be identity!"

    # Test Fk decomposition (same structure as z)
    Fk_batch = nlp.Fk_batch_func(z_batch, p_batch)
    Fk_stages = nlp.decompose_Fk_batch(Fk_batch)
    Fk_recomposed = nlp.compose_Fk_batch(Fk_stages)
    print(f"\nFk decomposition:")
    print(f"  Fk_batch shape: {Fk_batch.shape}")
    print(f"  Fk_stages shape: {Fk_stages.shape}")
    Fk_decomp_error = torch.norm(Fk_batch - Fk_recomposed)
    print(f"  Fk Decompose-Compose error: {Fk_decomp_error:.2e}")
    assert Fk_decomp_error < 1e-10, "Fk Decompose-Compose should be identity!"

    # Test pseudo-parameters
    print(f"\nPseudo-parameters:")
    print(f"  n_p_stage = {nlp.n_p_stage} (3*nx + nu = 3*{nlp.nx} + {nlp.nu})")
    p_stages = nlp.build_pseudo_params_batch(z_batch, p_batch)
    print(f"  p_stages shape: {p_stages.shape}")
    assert p_stages.shape == (n_batch, nlp.N, nlp.n_p_stage)
    
    # Verify structure with labeled test
    p_single = p_batch[0]
    p_stages_single = nlp.build_pseudo_params(z_labeled, p_single)
    print(f"\n  Labeled pseudo-params (p_k = [x_k, x_k2, u_k1, nu_k1]):")
    for k in range(min(N, 4)):
        p_k = p_stages_single[k]
        x_k_val = p_k[0].item() if k > 0 else "p"
        x_k2_val = p_k[nlp.nx].item()
        u_k1_val = p_k[2*nlp.nx].item()
        nu_k1_val = p_k[2*nlp.nx + nlp.nu].item()
        print(f"    k={k}: x_k={x_k_val}, x_k2={x_k2_val:.0f}, u_k1={u_k1_val:.0f}, nu_k1={nu_k1_val:.0f}")

    print("\nTime-Stage Decomposition test PASSED!")

    nlp.setup_for_batch(n_batch)
    z_batch = nlp.batch_gen_z(n_batch)
    p_batch = nlp.batch_gen_p(n_batch)
    Tk_vals_cusadi = nlp.Tk_batch_func(z_batch, p_batch)
    Tk_true = []
    for i in range(n_batch):
        Tk_i = nlp.Tk_nodiff_func_ca(z_batch[i, :].cpu().numpy(), p_batch[i, :].cpu().numpy())
        Tk_true.append(Tk_i)
    Tk_true = torch.tensor(np.array(Tk_true), dtype=nlp.dtype, device=nlp.device).squeeze()
    Tk_diff = torch.norm(Tk_vals_cusadi - Tk_true)
    print(f"Tk_diff_max = {Tk_diff.abs().max():.2e}")

    Fk_vals_cusadi = nlp.Fk_batch_func(z_batch, p_batch)
    Fk_true = []
    for i in range(n_batch):
        Fk_i = nlp.Fk_func_ca(z_batch[i, :].cpu().numpy(), p_batch[i, :].cpu().numpy())
        Fk_true.append(Fk_i)
    Fk_true = torch.tensor(np.array(Fk_true), dtype=nlp.dtype, device=nlp.device).squeeze()
    Fk_diff = torch.norm(Fk_vals_cusadi - Fk_true)
    print(f"Fk_diff_max = {Fk_diff.abs().max():.2e}")

    w_batch, nu_batch, lam_batch = nlp.extract_primal_dual_batch(z_batch)
    g_vals_cusadi = nlp.g_batch_func(w_batch, p_batch)
    g_true = []
    for i in range(n_batch):
        g_i = nlp.g_func_ca(w_batch[i, :].cpu().numpy(), p_batch[i, :].cpu().numpy())
        g_true.append(g_i)
    g_true = torch.tensor(np.array(g_true), dtype=nlp.dtype, device=nlp.device).squeeze()
    g_diff = torch.norm(g_vals_cusadi - g_true)
    print(f"g_diff_max = {g_diff.abs().max():.2e}")

    w_batch, nu_batch, lam_batch = nlp.extract_primal_dual_batch(z_batch)
    g_vals_cusadi = nlp.g_batch_func(w_batch, p_batch)
    g_true = []
    for i in range(n_batch):
        g_i = nlp.g_func_ca(w_batch[i, :].cpu().numpy(), p_batch[i, :].cpu().numpy())
        g_true.append(g_i)
    g_true = torch.tensor(np.array(g_true), dtype=nlp.dtype, device=nlp.device).squeeze()
    g_diff = torch.norm(g_vals_cusadi - g_true)
    print(f"g_diff_max = {g_diff.abs().max():.2e}")
