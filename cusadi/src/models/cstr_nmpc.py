"""
Economic NMPC for CSTR Cascade Model (Simplified, No Initial Value Embedding)

Standard Form NLP:
    min_w  f(w, p)
    s.t.   h(w, p) = 0          (equality: dynamics only)
           g(w, p) <= 0         (inequality: box bounds)
    
    where:
        w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]  (decision variables)
        p = x0  (parameters: initial state)
        
    The initial state x0 enters the dynamics directly at the first time step:
        h = [x_1 - F(x0, u_0); x_2 - F(x_1, u_1); ...; x_N - F(x_{N-1}, u_{N-1})]
"""

from dataclasses import dataclass, field, fields
import casadi as ca
import numpy as np


def _serialize_scalar(v):
    """Encode ±inf as a string for JSON compatibility."""
    if isinstance(v, float) and np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _deserialize_scalar(v):
    """Decode inf strings back to float."""
    if v == "inf":  return np.inf
    if v == "-inf": return -np.inf
    return v

from src.models.cstr_cascade import CSTRCascade, CSTRParams


@dataclass
class NMPCParams:
    """Parameters for the Economic/Tracking NMPC formulation.
    
    Scaling vs Bounds:
        - Scaling ranges (x_scaling/u_scaling): Fixed ranges for min-max normalization.
          All variables are scaled to [0, 1] using these ranges. These should not change.
        - Bounds (x_bounds/u_bounds): Decision variable bounds for the NLP.
          By default equal to scaling ranges, but can be different.
          Use +inf/-inf to disable specific bound constraints.
    """
    N: int = 10
    w_R: float = 10.0
    w_S: float = 1.0
    w_uA: float = 0.1
    w_uB: float = 0.0
    objective: str = "economic"  # "tracking" or "economic"
    setpoint_cR: float = None
    setpoint_cS: float = None
    print_level: int = 5
    max_iter: int = 5000
    ipopt_tol: float = 1e-10
    
    # Scaling ranges: dict with keys matching state/input names, values are (min, max) tuples
    # Used for min-max normalization to [0, 1]
    x_scaling: dict = field(default_factory=lambda: {
        "cA": (0.0, 0.1), # "cA": (0.0, 0.25),
        "cB": (0.0, 1.0), # "cB": (0.0, 4.0),
        "cR": (0.0, 4.0),
        "cS": (0.0, 1.0),
        "Tr": (20.0, 80.0),
    })
    u_scaling: dict = field(default_factory=lambda: {
        "uA": (0.0, 1.5),
        "uB": (0.0, 1.5),
    })
    
    # Decision variable bounds: dict with keys matching state/input names, values are (lb, ub) tuples
    # Use np.inf/-np.inf to disable bounds.
    # Example: {"cS": (0.0, 0.5)} to tighten cS bound, {"cA": (-np.inf, np.inf)} to remove cA bounds
    x_bounds: dict = field(default_factory=lambda: {
        "cA": (0.0, np.inf),
        "cB": (-np.inf, np.inf),
        "cR": (-np.inf, np.inf),
        "cS": (-np.inf, np.inf),
        "Tr": (20.0, 80.0),
    })
    u_bounds: dict = field(default_factory=lambda: {
        "uA": (0.0, 1.5),
        "uB": (0.0, 1.5),
    })

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict.

        Dict fields (x_scaling, u_scaling, x_bounds, u_bounds): tuples → lists, ±inf → strings.
        Scalar fields: passed through unchanged.
        """
        result = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                result[f.name] = {
                    k: [_serialize_scalar(t[0]), _serialize_scalar(t[1])]
                    for k, t in v.items()
                }
            else:
                result[f.name] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NMPCParams":
        """Reconstruct from a dict produced by to_dict().

        Converts lists back to tuples and restores ±inf from strings.
        """
        kwargs = {}
        for k, v in d.items():
            if isinstance(v, dict):
                kwargs[k] = {
                    kk: (_deserialize_scalar(vv[0]), _deserialize_scalar(vv[1]))
                    for kk, vv in v.items()
                }
            else:
                kwargs[k] = v
        return cls(**kwargs)


class CSTRNMPC:
    """Economic NMPC controller for the CSTR cascade model.
    
    No initial value embedding: x0 enters dynamics directly at first time step.
    """

    def __init__(self, model_params, nmpc_params=None):
        self.model_params = model_params
        self.nmpc_params = nmpc_params if nmpc_params is not None else NMPCParams()

        self.model = CSTRCascade(model_params, integrator="implicit_euler")

        self.nx = model_params.nx
        self.nu = model_params.nu
        self.N = self.nmpc_params.N

        self._build_scaling_ranges()
        self._build_bounds()
        self._build_nlp()
        self._build_solver()

    def _dict_to_arrays(self, d: dict, size: int, names_and_slices: list) -> tuple:
        """Convert {name: (lo, hi)} dict to (lo_array, hi_array)."""
        arr_lo = np.zeros(size)
        arr_hi = np.zeros(size)
        for name, (start, end) in names_and_slices:
            arr_lo[start:end], arr_hi[start:end] = d[name]
        return arr_lo, arr_hi

    def _build_scaling_ranges(self):
        """Build scaling ranges for min-max normalization."""
        nr = self.model_params.nr

        x_scaling = self.nmpc_params.x_scaling
        u_scaling = self.nmpc_params.u_scaling
        self.scaling_ranges = {**x_scaling, **u_scaling}

        # State scaling: names → array slices
        x_slices = [("cA", (0, nr)), ("cB", (nr, 2*nr)), ("cR", (2*nr, 3*nr)),
                    ("cS", (3*nr, 4*nr)), ("Tr", (4*nr, 5*nr))]
        self.x_min, self.x_max = self._dict_to_arrays(x_scaling, self.nx, x_slices)

        # Input scaling (Tj is now a fixed parameter, not a control input)
        u_slices = [("uA", (0, nr)), ("uB", (nr, 2*nr))]
        self.u_min, self.u_max = self._dict_to_arrays(u_scaling, self.nu, u_slices)

        # Compute scaling factors: scale = max - min
        self.x_scale = self.x_max - self.x_min
        self.u_scale = self.u_max - self.u_min
        self.x_scale[self.x_scale == 0.0] = 1.0
        self.u_scale[self.u_scale == 0.0] = 1.0

        # Build w_scale and w_offset for the structure: [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        w_scale_list = [self.u_scale]  # u_0
        w_offset_list = [self.u_min]
        for k in range(1, self.N):
            w_scale_list.append(self.x_scale)  # x_k
            w_scale_list.append(self.u_scale)  # u_k
            w_offset_list.append(self.x_min)
            w_offset_list.append(self.u_min)
        w_scale_list.append(self.x_scale)  # x_N
        w_offset_list.append(self.x_min)

        self.w_scale = np.concatenate(w_scale_list)
        self.w_offset = np.concatenate(w_offset_list)

    def _build_bounds(self):
        """Build decision variable bounds."""
        nr = self.model_params.nr
        
        x_bounds = self.nmpc_params.x_bounds
        u_bounds = self.nmpc_params.u_bounds
        self.bounds = {**x_bounds, **u_bounds}
        
        # State bounds
        x_slices = [("cA", (0, nr)), ("cB", (nr, 2*nr)), ("cR", (2*nr, 3*nr)),
                    ("cS", (3*nr, 4*nr)), ("Tr", (4*nr, 5*nr))]
        self.x_lb, self.x_ub = self._dict_to_arrays(x_bounds, self.nx, x_slices)
        
        # Input bounds (Tj is now a fixed parameter, not a control input)
        u_slices = [("uA", (0, nr)), ("uB", (nr, 2*nr))]
        self.u_lb, self.u_ub = self._dict_to_arrays(u_bounds, self.nu, u_slices)

    def _unscale_x(self, x_scaled):
        """Unscale state from [0,1] to physical units: x = x_scaled * x_scale + x_min"""
        return x_scaled * self.x_scale + self.x_min
    
    def _unscale_u(self, u_scaled):
        """Unscale input from [0,1] to physical units: u = u_scaled * u_scale + u_min"""
        return u_scaled * self.u_scale + self.u_min
    
    def _scale_x(self, x):
        """Scale state to [0,1]: x_scaled = (x - x_min) / x_scale"""
        return (x - self.x_min) / self.x_scale
    
    def _scale_u(self, u):
        """Scale input to [0,1]: u_scaled = (u - u_min) / u_scale"""
        return (u - self.u_min) / self.u_scale

    def _x_idx(self, k: int) -> tuple:
        """Index range for x_k in w (k >= 1). Returns (start, end)."""
        # Structure: [u_0, x_1, u_1, x_2, u_2, ..., x_{N-1}, u_{N-1}, x_N]
        # x_k starts after: u_0 + (k-1) * (nx + nu)
        start = self.nu + (k - 1) * (self.nx + self.nu)
        return start, start + self.nx

    def _u_idx(self, k: int) -> tuple:
        """Index range for u_k in w (k >= 0). Returns (start, end)."""
        # u_0 at 0, then u_k starts after: u_0 + k * (nx + nu) - nx = nu + (k-1)*(nx+nu) + nx
        if k == 0:
            return 0, self.nu
        start = self.nu + (k - 1) * (self.nx + self.nu) + self.nx
        return start, start + self.nu

    def _get_last_reactor_outputs(self, x_scaled):
        """Return scaled [0, 1] values of (cR, cS) for the last reactor.

        Args:
            x_scaled: Slice of the scaled state vector for one time step.
        """
        nr = self.model_params.nr
        cR = x_scaled[3*nr - 1]
        cS = x_scaled[4*nr - 1]
        return cR, cS

    def _build_economic_cost(self, w):
        """Build economic cost: maximize c_R, minimize c_S of last reactor."""
        J = ca.SX(0.0)
        w_R = self.nmpc_params.w_R
        w_S = self.nmpc_params.w_S
        N = self.N

        # Stage costs for k=1,...,N
        for k in range(0, N):
            i_start, i_end = self._x_idx(k+1)
            cR, cS = self._get_last_reactor_outputs(w[i_start:i_end])
            J += -w_R * cR + w_S * cS

        # Control effort costs for k=0,...,N-1 (scaled [0, 1] inputs)
        for k in range(0,N):
            u_start, u_end = self._u_idx(k)
            u_k = w[u_start:u_end]
            uA_k = u_k[0:self.model_params.nr]
            uB_k = u_k[self.model_params.nr:]
            J += self.nmpc_params.w_uA * ca.sumsqr(uA_k) + self.nmpc_params.w_uB * ca.sumsqr(uB_k)

        return J

    def _build_tracking_objective(self, w):
        """Build tracking cost: track setpoints for c_R and c_S."""
        J = ca.SX(0.0)
        w_R = self.nmpc_params.w_R
        w_S = self.nmpc_params.w_S
        setpoint_cR = self.nmpc_params.setpoint_cR
        setpoint_cS = self.nmpc_params.setpoint_cS
        N = self.N

        cR_scale = self.x_scale[3*self.model_params.nr - 1]
        cS_scale = self.x_scale[4*self.model_params.nr - 1]
        cR_min   = self.x_min[3*self.model_params.nr - 1]
        cS_min   = self.x_min[4*self.model_params.nr - 1]

        # Stage costs for k=1,...,N
        for k in range(0, N):
            i_start, i_end = self._x_idx(k+1)
            cR, cS = self._get_last_reactor_outputs(w[i_start:i_end])
            J += w_R * (cR - (setpoint_cR - cR_min)/cR_scale)**2 \
               + w_S * (cS - (setpoint_cS - cS_min)/cS_scale)**2

        # Control effort costs for k=0,...,N-1 (scaled [0, 1] inputs)
        for k in range(0,N):
            u_start, u_end = self._u_idx(k)
            u_k = w[u_start:u_end]
            uA_k = u_k[0:self.model_params.nr]
            uB_k = u_k[self.model_params.nr:]
            J += self.nmpc_params.w_uA * ca.sumsqr(uA_k) + self.nmpc_params.w_uB * ca.sumsqr(uB_k)

        return J

    def _build_nlp(self):
        """Build the NLP in standard form.
        
        Decision variables: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Parameters: p = x0 (initial state, scaled to [0,1])
        
        All variables in w are scaled to [0, 1] using min-max normalization.
        """
        N = self.N
        n_x = self.nx
        n_u = self.nu

        w_list = []
        # States and controls for k=0,...,N-1 (controls) and k=1,...,N (states)
        for k in range(0, N):
            u_k = ca.SX.sym(f"u_{k}", n_u)
            w_list.append(u_k)

            x_k = ca.SX.sym(f"x_{k+1}", n_x)
            w_list.append(x_k)
        # Stack decision variables
        w = ca.vertcat(*w_list)
        n_w = w.size()[0]

        # default initial guess
        w0 = []
        x0_default_scaled = self._scale_x(self.model_params.get_default_x0())
        for k in range(0, N):
            # u_k initial guess: mid of scaling range
            u0_k = 0.5 * np.ones(n_u)
            w0.extend(u0_k.tolist())

            # x_{k+1} initial guess: default x0
            w0.extend(x0_default_scaled.tolist())
        w0 = np.array(w0)

        # Parameters: x0 (initial state, scaled to [0,1])
        x0_param = ca.SX.sym("x0", n_x)
        p = x0_param
        n_p = p.size()[0]

        # Objective
        if self.nmpc_params.objective == "economic":
            f = self._build_economic_cost(w)
        elif self.nmpc_params.objective == "tracking":
            f = self._build_tracking_objective(w)
        else:
            raise ValueError(f"Unknown objective type: {self.nmpc_params.objective}")

        # Equality constraints h(w, p) = 0 (dynamics only)
        h_list = []
        dt = self.model_params.dt

        # First dynamics constraint: x_1 - F(x0_param, u_0) = 0
        u_0_s, u_0_e = self._u_idx(0)
        x_1_s, x_1_e = self._x_idx(1)
        u_0 = w[u_0_s:u_0_e]
        x_1 = w[x_1_s:x_1_e]
        
        f_eval = self.model.f_ode(
            x_1 * self.x_scale + self.x_min,
            u_0 * self.u_scale + self.u_min
        )
        # h_list.append(x_1 * self.x_scale - x0_param * self.x_scale - dt * f_eval) # Note: x0_param is scaled, x_1 = x0param + dt * f(x_1, u_0)
        h_list.append(x_1 - x0_param - dt * f_eval/self.x_scale) # scaled dynamics

        # Dynamics constraints for k=1,...,N-1
        for k in range(1, N):
            x_k_s, x_k_e = self._x_idx(k)
            u_k_s, u_k_e = self._u_idx(k)
            x_kp1_s, x_kp1_e = self._x_idx(k + 1)
            
            x_k = w[x_k_s:x_k_e]
            u_k = w[u_k_s:u_k_e]
            x_k_next = w[x_kp1_s:x_kp1_e]

            f_eval = self.model.f_ode(
                x_k_next * self.x_scale + self.x_min,
                u_k * self.u_scale + self.u_min
            )
            # h_list.append(x_k_next * self.x_scale - x_k * self.x_scale - dt * f_eval)
            h_list.append(x_k_next - x_k - dt * f_eval/self.x_scale)

        h = ca.vertcat(*h_list)
        n_h = h.size()[0]

        # Inequality constraints g(w, p) <= 0 (from box bounds)
        g_list = []
        
        # Build bound vectors matching w structure
        lbw_list = [self.u_lb]  # u_0
        ubw_list = [self.u_ub]
        for k in range(1, N):
            lbw_list.append(self.x_lb)
            ubw_list.append(self.x_ub)
            lbw_list.append(self.u_lb)
            ubw_list.append(self.u_ub)
        lbw_list.append(self.x_lb)  # x_N
        ubw_list.append(self.x_ub)
        
        lbw = np.concatenate(lbw_list)
        ubw = np.concatenate(ubw_list)
        
        # Upper bounds: w_phys - ub <= 0
        for i in range(n_w):
            if np.isfinite(ubw[i]):
                # g_list.append(w[i] * self.w_scale[i] + self.w_offset[i] - ubw[i])
                g_list.append(w[i] + self.w_offset[i]/self.w_scale[i] - ubw[i]/self.w_scale[i])

        # Lower bounds: lb - w_phys <= 0
        for i in range(n_w):
            if np.isfinite(lbw[i]):
                # g_list.append(lbw[i] - (w[i] * self.w_scale[i] + self.w_offset[i]))
                g_list.append(lbw[i]/self.w_scale[i]  - (w[i] + self.w_offset[i]/self.w_scale[i]))

        if len(g_list) > 0:
            g = ca.vertcat(*g_list)
            n_g = g.size()[0]
        else:
            g = ca.SX(0, 1)
            n_g = 0

        self.nlp = {
            "w": w,
            "p": p,
            "f": f,
            "h": h,
            "g": g,
            "n_w": int(n_w),
            "n_p": int(n_p),
            "n_h": int(n_h),
            "n_g": int(n_g),
            "n_z": int(n_w + n_h + n_g),
            "w0": w0,
            "n_x": n_x,
            "n_u": n_u,
            "N": N,
        }



    def _build_solver(self):
        """Build IPOPT solver."""
        n_h = self.nlp["n_h"]
        n_g = self.nlp["n_g"]
        w = self.nlp["w"]
        p = self.nlp["p"]
        f = self.nlp["f"]
        h = self.nlp["h"]
        g = self.nlp["g"]
        w0 = self.nlp["w0"]

        g_solver_list = []
        lbg_solver = []
        ubg_solver = []

        if n_h > 0:
            g_solver_list.append(h)
            lbg_solver.extend([0.0] * n_h)
            ubg_solver.extend([0.0] * n_h)

        if n_g > 0:
            g_solver_list.append(g)
            lbg_solver.extend([-np.inf] * n_g)
            ubg_solver.extend([0.0] * n_g)

        if len(g_solver_list) > 0:
            g_solver = ca.vertcat(*g_solver_list)
        else:
            g_solver = ca.SX(0, 1)

        lbg_solver = np.array(lbg_solver)
        ubg_solver = np.array(ubg_solver)

        solver_nlp = {"x": w, "f": f, "g": g_solver, "p": p}
        opts = {
            "ipopt.print_level": self.nmpc_params.print_level,
            "ipopt.max_iter": self.nmpc_params.max_iter,
            "ipopt.tol": self.nmpc_params.ipopt_tol,
            # "ipopt.mu_target": 5*1e-5,
            "print_time": False,
        }
        self.solver = ca.nlpsol("nmpc_solver", "ipopt", solver_nlp, opts)

        solver_nlp.update({"w0": w0, "lbg": lbg_solver, "ubg": ubg_solver})
        self.solver_nlp = solver_nlp

    def extract_states_controls(self, w):
        """Extract states X and controls U from decision vector w.
        
        Note: x_0 is not in w; caller should prepend x0 if full trajectory needed.
        Returns:
            X: States for x_1,...,x_N ((N)*nx,)
            U: Controls for u_0,...,u_{N-1} (N*nu,)
        """
        X_list = []
        U_list = []

        # States x_1, ..., x_N
        for k in range(1, self.N + 1):
            i_start, i_end = self._x_idx(k)
            X_list.append(w[i_start:i_end])

        # Controls u_0, ..., u_{N-1}
        for k in range(self.N):
            i_start, i_end = self._u_idx(k)
            U_list.append(w[i_start:i_end])

        X = np.concatenate(X_list)
        U = np.concatenate(U_list)
        return X, U

    def stack_states_controls(self, X, U):
        """Stack states X and controls U into decision vector w.
        
        Args:
            X: States for x_1,...,x_N (N*nx,) or (N, nx)
            U: Controls for u_0,...,u_{N-1} (N*nu,) or (N, nu)
        """
        N = self.N
        n_x = self.nx
        n_u = self.nu

        X = X.flatten()
        U = U.flatten()

        # Structure: u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N
        w_parts = [U[:n_u]]  # u_0
        for k in range(1, N):
            w_parts.append(X[(k-1)*n_x : k*n_x])  # x_k
            w_parts.append(U[k*n_u : (k+1)*n_u])  # u_k
        w_parts.append(X[(N-1)*n_x : N*n_x])  # x_N

        return np.concatenate(w_parts)

    def solve(self, x0):
        """Solve the NMPC problem.
        
        Args:
            x0: Current state (nx,), scaled.
            return_primal_dual: If True, return full primal-dual solution vector z_opt.
        
        Returns:
            z_opt: Optimal primal-dual solution vector [w_opt, nu_opt, lam_opt], primal variables scaled
            info: Solver information dict
        """
        import time

        x0 = np.atleast_1d(x0).flatten()
        p = x0

        w0 = self.solver_nlp["w0"]
        lbg = self.solver_nlp["lbg"]
        ubg = self.solver_nlp["ubg"]

        t_start = time.perf_counter()
        sol = self.solver(x0=w0, lbg=lbg, ubg=ubg, p=p)
        t_solve = time.perf_counter() - t_start

        w_opt = np.array(sol["x"]).flatten()
        lam_g_opt = np.array(sol["lam_g"]).flatten()
        obj_val = float(sol["f"])

        stats = self.solver.stats()
        info = {
            "objective": obj_val,
            "iterations": stats.get("iter_count", -1),
            "success": stats.get("success", False),
            "return_status": stats.get("return_status", "unknown"),
            "solve_time": t_solve,
        }

        n_h = self.nlp["n_h"]
        n_g = self.nlp["n_g"]
        nu_opt = np.array(lam_g_opt[:n_h]) if n_h > 0 else np.array([])
        lam_opt = np.array(lam_g_opt[n_h:n_h + n_g]) if n_g > 0 else np.array([])
        z_opt = np.hstack([w_opt, nu_opt, lam_opt])

        return z_opt, info

    def mpc_step(self, x0_unscaled):
        """Execute one MPC step: scale input, solve, extract and unscale control action.
        
        Args:
            x0_unscaled: Current state in physical units (nx,).
        
        Returns:
            u0: First optimal control action in physical units (nu,).
            info: Solver information dict.
        """
        x0_unscaled = np.atleast_1d(x0_unscaled).flatten()
        
        # Scale initial state
        x0_scaled = self._scale_x(x0_unscaled)
        
        # Solve NMPC
        z_opt, info = self.solve(x0_scaled)

        # Extract w_opt
        w_opt = z_opt[:self.nlp["n_w"]]
        
        # Extract x_1,...,x_N and u_0,...,u_{N-1}
        _, U_opt = self.extract_states_controls(w_opt)
        
        # # Unscale states (X_pred has N*nx elements for x_1,...,x_N)
        # X_pred_unscaled = X_pred.reshape((self.N, self.nx)) * self.x_scale + self.x_min
        
        # # Prepend x0 to state trajectory (x0_unscaled is already in physical units from input)
        # x_opt = np.vstack([x0_unscaled.reshape(1, -1), X_pred_unscaled])
        
        # Unscale controls
        U_opt_unscaled = U_opt.reshape((self.N, self.nu)) * self.u_scale + self.u_min
        u_opt = U_opt_unscaled

        return u_opt[0], info
        
    def simulate_closed_loop(self, x0, n_sim, verbose=True):
        """Run closed-loop MPC simulation.
        
        Args:
            x0: Initial state (nx,), scaled.
            n_sim: Number of simulation steps.
        """
        x0 = np.atleast_1d(x0).flatten()

        x_traj = np.zeros((n_sim + 1, self.nx))
        u_traj = np.zeros((n_sim, self.nu))
        obj_vals = np.zeros(n_sim)
        solve_times = np.zeros(n_sim)

        x_traj[0] = x0 * self.x_scale + self.x_min  # Unscale

        for k in range(n_sim):
            if verbose and k % 10 == 0:
                print(f"  Step {k}/{n_sim}...")

            # Scale current state for solver
            u_opt, info = self.mpc_step(x_traj[k])

            if not info["success"]:
                print(f"Warning: NMPC solver failed at step {k}: {info['return_status']}")

            u_traj[k] = u_opt
            obj_vals[k] = info["objective"]
            solve_times[k] = info["solve_time"]

            x_traj[k + 1] = self.model.simulate_step(x_traj[k], u_traj[k])

        if verbose:
            print(f"  Completed {n_sim} steps.")
            print(f"  Mean solve time: {solve_times.mean() * 1000:.2f} ms")
            print(f"  Max solve time: {solve_times.max() * 1000:.2f} ms")

        return {
            "x": x_traj,
            "u": u_traj,
            "obj": obj_vals,
            "solve_time": solve_times,
        }

    def get_dimensions(self):
        """Get problem dimensions."""
        return {
            "nx": self.nx,
            "nu": self.nu,
            "N": self.N,
            "n_w": self.nlp["n_w"],
            "n_h": self.nlp["n_h"],
            "n_g": self.nlp["n_g"],
            "n_z": self.nlp["n_z"],
            "n_p": self.nlp["n_p"],
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Economic NMPC Test (No Initial Value Embedding)")
    print("=" * 60)

    nr = 5
    dt = 1.0
    N = 20
    n_sim = 100

    model_params = CSTRParams(nr=nr, dt=dt)
    nmpc_params = NMPCParams(
        N=N
    )
    # nmpc_params.x_bounds["cA"] = (-np.inf, np.inf)
    # nmpc_params.x_bounds["cA"] = (0.0, np.inf)
    # nmpc_params.x_bounds["cB"] = (-np.inf, np.inf)
    # nmpc_params.x_bounds["cR"] = (-np.inf, np.inf)
    # nmpc_params.x_bounds["cS"] = (-np.inf, np.inf)
    # nmpc_params.x_bounds["Tr"] = (-np.inf, np.inf)  # Tighten Tr bounds for testing

    print(f"\nModel: {nr} reactors, dt={dt}s, fixed Tj={model_params.Tj}°C")
    print(f"Economic NMPC: N={N}, w_R={nmpc_params.w_R}, w_S={nmpc_params.w_S}")

    print("\nBuilding NMPC...")
    nmpc = CSTRNMPC(model_params, nmpc_params)

    dims = nmpc.get_dimensions()
    print(f"  States: nx={dims['nx']}")
    print(f"  Inputs: nu={dims['nu']}")
    print(f"  Decision variables: {dims['n_w']}")
    print(f"  Equality constraints: {dims['n_h']}")
    print(f"  Inequality constraints: {dims['n_g']}")

    print("\nRunning closed-loop simulation...")
    x0 = nmpc._scale_x(model_params.get_default_x0())

    results = nmpc.simulate_closed_loop(x0, n_sim, verbose=True)

    def add_bounds_to_plot(ax, nmpc, key):
        """Add bound lines to the plot if bounds are finite."""
        if nmpc.bounds[key][0] > -np.inf:
            ax.axhline(nmpc.bounds[key][0], color="r", linestyle="--", alpha=0.5, label=f"lb_{key}")
        if nmpc.bounds[key][1] < np.inf:
            ax.axhline(nmpc.bounds[key][1], color="r", linestyle="--", alpha=0.5, label=f"ub_{key}")

    def add_lim_to_plot(ax, nmpc, key):
        lims = []
        if nmpc.bounds[key][0] > -np.inf:
            lims.append(nmpc.bounds[key][0])
        else:
            lims.append(nmpc.scaling_ranges[key][0])
        if nmpc.bounds[key][1] < np.inf:
            lims.append(nmpc.bounds[key][1])
        else:
            lims.append(nmpc.scaling_ranges[key][1])
        offset = 0.1 * (max(lims) - min(lims))
        ax.set_ylim(min(lims) - offset, max(lims) + offset)

    def add_setpoint_to_plot(ax, nmpc, key, setpoint):
        """Add setpoint line to the plot."""
        ax.axhline(setpoint, color="g", linestyle="--", alpha=0.5, label=f"setpoint_{key}")

    # Plotting
    t = np.arange(n_sim + 1) * dt
    t_u = t[:-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{nmpc_params.objective.capitalize()} NMPC ({nr} Reactors, N={N}, Tj={model_params.Tj}°C)", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, nr))

    ax = axes[0, 0]
    for i in range(nr):
        ax.plot(t, results["x"][:, i], color=colors[i], label=f"cA_{i+1}")
    ax.set_ylabel("$c_A$ [mol/L]")
    add_lim_to_plot(ax, nmpc, "cA")
    add_bounds_to_plot(ax, nmpc, "cA")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i in range(nr):
        ax.plot(t, results["x"][:, nr + i], color=colors[i], label=f"cB_{i+1}")
    ax.set_ylabel("$c_B$ [mol/L]")
    add_lim_to_plot(ax, nmpc, "cB")
    add_bounds_to_plot(ax, nmpc, "cB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i in range(nr):
        ax.plot(t, results["x"][:, 2*nr + i], color=colors[i], label=f"cR_{i+1}")
    ax.set_ylabel("$c_R$ [mol/L]")
    if nmpc_params.objective == "tracking":
        add_setpoint_to_plot(ax, nmpc, "cR", nmpc_params.setpoint_cR)
    add_lim_to_plot(ax, nmpc, "cR")
    add_bounds_to_plot(ax, nmpc,"cR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i in range(nr):
        ax.plot(t, results["x"][:, 3*nr + i], color=colors[i], label=f"cS_{i+1}")
    ax.set_ylabel("$c_S$ [mol/L]")
    if nmpc_params.objective == "tracking":
        add_setpoint_to_plot(ax, nmpc, "cS", nmpc_params.setpoint_cS)
    add_lim_to_plot(ax, nmpc, "cS")
    add_bounds_to_plot(ax, nmpc, "cS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle(f"Reactor Temperatures ({nmpc_params.objective.capitalize()} NMPC, {nr} Reactors)", fontsize=14)
    for i in range(nr):
        ax.plot(t, results["x"][:, 4*nr + i], color=colors[i], label=f"Tr_{i+1}")
    # Show fixed Tj as horizontal line
    ax.axhline(model_params.Tj, color="gray", linestyle="--", alpha=0.6, label="Tj (fixed)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    add_lim_to_plot(ax, nmpc, "Tr")
    add_bounds_to_plot(ax, nmpc, "Tr")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # New figure for control actions
    fig_controls, axes_controls = plt.subplots(1, 2, figsize=(14, 5))
    fig_controls.suptitle(f"Control Actions ({nmpc_params.objective.capitalize()} NMPC, {nr} Reactors)", fontsize=14)

    # uA subplot
    ax = axes_controls[0]
    for i in range(nr):
        u_extended = np.append(results["u"][:, i], results["u"][-1, i])
        t_u_extended = np.append(t_u, t_u[-1] + dt)
        ax.step(t_u_extended, u_extended, where="post", color=colors[i], label=f"uA_{i+1}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("uA [mol/s]")
    ax.set_title("Feed rate A")
    add_lim_to_plot(ax, nmpc, "uA")
    add_bounds_to_plot(ax, nmpc, "uA")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # uB subplot
    ax = axes_controls[1]
    for i in range(nr):
        u_extended = np.append(results["u"][:, nr + i], results["u"][-1, nr + i])
        t_u_extended = np.append(t_u, t_u[-1] + dt)
        ax.step(t_u_extended, u_extended, where="post", color=colors[i], label=f"uB_{i+1}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("uB [mol/s]")
    ax.set_title("Feed rate B")
    add_lim_to_plot(ax, nmpc, "uB")
    add_bounds_to_plot(ax, nmpc, "uB")
    ax.legend()
    ax.grid(True, alpha=0.3)


    plt.tight_layout() 
    plt.show()
