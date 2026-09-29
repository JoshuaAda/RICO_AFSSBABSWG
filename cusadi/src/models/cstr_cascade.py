"""Simplified CSTR Cascade Model for LISCO-Decomposed

A minimal CSTR cascade model for simulation and initial time decomposition experiments.
This version removes uncertainty handling and MPC-specific code for clarity.

The model implements a cascade of nr reactors with:
- States (per reactor): cA, cB, cR, cS (concentrations) and Tr (temperature)
- Inputs (per reactor): uA, uB (feed rates)
- Fixed parameter: Tj (jacket temperature, same for all reactors)

Reactions:
- A + B -> R (desired product), rate k1 * cA * cB
- 2A -> S (undesired side product), rate k2 * cA^2

Flow Structure:
    The first reactor receives an implicit solvent feed stream at volumetric rate
    V_out [L/s] with zero concentrations (cA=cB=cR=cS=0) and temperature Tr_in.
    Pure components A and B are added at molar rates uA, uB [mol/s]. This maintains
    constant volume in all reactors.

    Schematic:
    
                        uA[0], uB[0] (mol/s)
                             |
        Solvent feed    +---------+         +---------+
        (V_out, Tr_in)  | Reactor |  V_out  | Reactor |  V_out
        cA=cB=cR=cS=0 ->|    1    |-------->|    2    |--------> ...
                        |         |         |         |
                        +----+----+         +----+----+
                             |                   |
                           Tj[0]               Tj[1]

Unit System:
    The model uses Liters as the base volume unit, making parameter values
    physically consistent:

    | Parameter | Value | Unit        | Interpretation                    |
    |-----------|-------|-------------|-----------------------------------|
    | V_R       | 15    | L           | Total reactor volume              |
    | V_out     | 1     | L/s         | Volumetric flow rate              |
    | rho       | 1     | kg/L        | Density (= 1000 kg/m³ for water)  |
    | cp        | 0.5   | kJ/(kg·K)   | Heat capacity (organic solvent)   |
    | kA        | 2     | kW/K        | Heat transfer coeff. × area       |
    | uA, uB    | -     | mol/s       | Molar feed rates                  |
    | cA, cB... | -     | mol/L       | Concentrations                    |
    | Tr, Tj    | -     | °C          | Temperatures                      |

Note on Activation Energies:
    The activation energies (Ea1=500, Ea2=600 J/mol) are intentionally ~100x
    lower than realistic values (typically 40,000-80,000 J/mol). This makes
    the reactions nearly temperature-independent, which simplifies the dynamics
    but is physically unrealistic. This is acceptable for solver benchmarking
    where the focus is on the optimization algorithm, not physical accuracy.
"""
# TODO: adjust activation energies also in docstring

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

import casadi as ca
import numpy as np


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class CSTRParams:
    """Parameters for the simplified CSTR cascade model.

    The model uses Liters as the base volume unit. With rho=1 kg/L, this
    corresponds to a density of 1000 kg/m³ (water/aqueous solution).

    Attributes:
        nr: Number of reactors in the cascade.
        dt: Discretization time step [s].
        V_R: Total reactor volume [L].
        V_out: Volumetric outflow rate [L/s].
        k1: Reaction rate constant for A + B -> R [L/(mol·s)].
        k2: Reaction rate constant for 2A -> S [L/(mol·s)].
        delH1: Reaction enthalpy for reaction 1 [kJ/mol] (negative = exothermic).
        delH2: Reaction enthalpy for reaction 2 [kJ/mol] (negative = exothermic).
        Tr_in: Inlet temperature of solvent feed [°C].
        kA: Heat transfer coefficient × area [kW/K].
        rho: Density [kg/L] (1 kg/L = 1000 kg/m³).
        cp: Heat capacity [kJ/(kg·K)].
        Ea1: Activation energy for reaction 1 [J/mol] (intentionally low).
        Ea2: Activation energy for reaction 2 [J/mol] (intentionally low).
        R_gas: Universal gas constant [J/(mol·K)].
    """

    # Cascade structure
    nr: int = 1

    # Discretization
    dt: float = 1.0  # [s]

    # Physical parameters
    V_R: float = 15.0  # [L] Total reactor volume
    V_out: float = 1.0  # [L/s] Volumetric outflow rate
    k1: float = 2.0  # [L/(mol·s)] Reaction rate constant for A + B -> R
    k2: float = 2.0  # [L/(mol·s)] Reaction rate constant for 2A -> S
    delH1: float = -100.0  # [kJ/mol] Reaction enthalpy (exothermic)
    delH2: float = -50.0  # [kJ/mol] Reaction enthalpy (exothermic)
    Tr_in: float = 60.0  # [°C] Inlet temperature
    kA: float = 2.0  # [kW/K] Heat transfer coefficient * area
    rho: float = 1.0  # [kg/L] Density (= 1000 kg/m³)
    cp: float = 0.5  # [kJ/(kg·K)] Heat capacity
    Ea1: float = 500.0  # [J/mol] Activation energy for reaction 1 # TODO: check if these values are phyically reasonable
    Ea2: float = 600.0  # [J/mol] Activation energy for reaction 2 # TODO: check if these values are phyically reasonable
    R_gas: float = 8.3145  # [J/(mol·K)] Universal gas constant
    Tj: float = 50.0  # [°C] Fixed jacket temperature (same for all reactors)

    # Derived quantities (computed in __post_init__)
    V_i: float = field(init=False)  # [L] Volume per reactor
    nx: int = field(init=False)  # Total state dimension
    nu: int = field(init=False)  # Total input dimension

    def __post_init__(self) -> None:
        """Compute derived quantities and validate parameters."""
        self.V_i = self.V_R / self.nr
        self.nx = 5 * self.nr  # 5 states per reactor: cA, cB, cR, cS, Tr
        self.nu = 2 * self.nr  # 2 inputs per reactor: uA, uB (Tj is fixed)

    def to_dict(self) -> dict:
        """Serialize init fields to a plain dict (excludes derived V_i, nx, nu)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "CSTRParams":
        """Reconstruct a CSTRParams from a dict produced by to_dict()."""
        return cls(**d)

    def get_default_x0(self) -> np.ndarray:
        """Get default initial state (near-zero concentrations, inlet temperature)."""
        x0 = 1e-3 * np.ones(self.nx)
        x0[4 * self.nr :] = self.Tr_in  # Initial temperature = inlet temperature
        return x0

    def get_default_u0(self) -> np.ndarray:
        """Get default initial input (zeros)."""
        return np.zeros(self.nu)


# =============================================================================
# CSTR Cascade Model Class
# =============================================================================


class CSTRCascade:
    """Simplified CSTR Cascade model for simulation.

    This class provides:
    - Continuous-time ODE dynamics (CasADi symbolic)
    - Discrete-time integrator (CVODES or RK4)
    - Trajectory simulation

    Example:
        >>> model = CSTRCascade(CSTRParams(nr=3))
        >>> x0 = model.params.get_default_x0()
        >>> u = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 50, 50, 50])
        >>> x_next = model.simulate_step(x0, u)
    """

    def __init__(
        self,
        params: CSTRParams | None = None,
        integrator: Literal["cvodes", "rk4", "explicit_euler", "implicit_euler"] = "implicit_euler",
    ) -> None:
        """Initialize the CSTR cascade model.

        Args:
            params: Model parameters. If None, uses defaults.
            integrator: Integration method ('cvodes' or 'rk4').
        """
        self.params = params if params is not None else CSTRParams()
        self._integrator_method = integrator

        if integrator in ["rk4", "explicit_euler"]:

            # Validate dt relative to residence time for numerical stability
            tau = self.params.V_i / self.params.V_out  # residence time per reactor [s]
            dt_ratio = self.params.dt / tau
            max_ratio = 0.15  # 15% of residence time

            if dt_ratio > max_ratio:
                raise ValueError(
                    f"Time step dt={self.params.dt}s is too large for nr={self.params.nr} reactors. "
                    f"Residence time per reactor is tau={tau:.2f}s, giving dt/tau={dt_ratio:.1%}. "
                    f"For numerical stability with RK4 or explicit Euler integration, dt/tau should be <={max_ratio:.0%}. "
                    f"Recommended: dt <= {max_ratio * tau:.2f}s (e.g., dt={0.1 * tau:.2f}s for 10% margin)."
                )


        # Build CasADi functions
        self._build_symbolic_variables()
        self._build_dynamics()
        self._build_integrator(integrator)

    # -------------------------------------------------------------------------
    # Symbolic Variables
    # -------------------------------------------------------------------------

    def _build_symbolic_variables(self) -> None:
        """Create symbolic variables for states and inputs."""
        nr = self.params.nr

        # States (per reactor, then stacked)
        self.sym_cA = ca.SX.sym("cA", nr)
        self.sym_cB = ca.SX.sym("cB", nr)
        self.sym_cR = ca.SX.sym("cR", nr)
        self.sym_cS = ca.SX.sym("cS", nr)
        self.sym_Tr = ca.SX.sym("Tr", nr)
        self.sym_x = ca.vertcat(
            self.sym_cA, self.sym_cB, self.sym_cR, self.sym_cS, self.sym_Tr
        )

        # Inputs (per reactor, then stacked)
        self.sym_uA = ca.SX.sym("uA", nr)
        self.sym_uB = ca.SX.sym("uB", nr)
        self.sym_u = ca.vertcat(self.sym_uA, self.sym_uB)

    # -------------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------------

    def _build_dynamics(self) -> None:
        """Build the continuous-time ODE dynamics as a CasADi function.

        Flow structure:
        - First reactor receives an implicit solvent feed at rate V_out [L/s]
          with zero concentrations and temperature Tr_in. Pure components A and B
          are added at molar rates uA, uB [mol/s].
        - Subsequent reactors receive the effluent from the previous reactor.
        - All reactors have constant volume V_i = V_R / nr.
        """
        p = self.params
        nr = p.nr

        # Temperature in Kelvin
        # T_kelvin = self.sym_Tr + 273.15

        # With a safeguarded version using soft clipping:
        T_kelvin_raw = self.sym_Tr + 273.15
        # T_kelvin = self.sym_Tr + 273.15
        T_min = 150.0  # Minimum temperature [K]

        # Soft lower bound using softplus-like function
        # This is smooth and differentiable, unlike ca.fmax
        # T_kelvin = T_min + ca.log(1.0 + ca.exp(T_kelvin_raw - T_min))

        # Alternative version (may cause issues in optimization)
        T_kelvin = T_min + ca.fmax(0, T_kelvin_raw - T_min) # + ca.log1p(ca.exp(-ca.fabs(T_kelvin_raw - T_min))) # TODO: check if this is really a good idea (might be problematic for solver learning)

        # Reaction rates with Arrhenius kinetics
        # k_eff = k * exp(-Ea / (R * T))
        k1_eff = p.k1 * ca.exp(-p.Ea1 / (p.R_gas * T_kelvin))
        k2_eff = p.k2 * ca.exp(-p.Ea2 / (p.R_gas * T_kelvin))

        # Reaction rates [mol/(L·s)]
        r1 = k1_eff * self.sym_cA * self.sym_cB  # A + B -> R
        r2 = k2_eff * self.sym_cA**2  # 2A -> S

        # Heat generation coefficients
        # Q = -delH / (rho * cp) * k
        Q1 = -p.delH1 / (p.rho * p.cp) * p.k1
        Q2 = -p.delH2 / (p.rho * p.cp) * p.k2

        # Build ODEs reactor by reactor
        rhs_cA = []
        rhs_cB = []
        rhs_cR = []
        rhs_cS = []
        rhs_Tr = []

        for i in range(nr):
            # Flow terms
            if i == 0:
                # First reactor: outflow only (feed via uA, uB)
                flow_cA = -p.V_out / p.V_i * self.sym_cA[i]
                flow_cB = -p.V_out / p.V_i * self.sym_cB[i]
                flow_cR = -p.V_out / p.V_i * self.sym_cR[i]
                flow_cS = -p.V_out / p.V_i * self.sym_cS[i]
                flow_Tr = p.V_out / p.V_i * (p.Tr_in - self.sym_Tr[i])
            else:
                # Subsequent reactors: inflow from previous reactor
                flow_cA = p.V_out / p.V_i * (self.sym_cA[i - 1] - self.sym_cA[i])
                flow_cB = p.V_out / p.V_i * (self.sym_cB[i - 1] - self.sym_cB[i])
                flow_cR = p.V_out / p.V_i * (self.sym_cR[i - 1] - self.sym_cR[i])
                flow_cS = p.V_out / p.V_i * (self.sym_cS[i - 1] - self.sym_cS[i])
                flow_Tr = p.V_out / p.V_i * (self.sym_Tr[i - 1] - self.sym_Tr[i])

            # Concentration ODEs
            # dcA/dt = flow - r1 - 2*r2 + feed_A/V
            rhs_cA.append(flow_cA - r1[i] - 2 * r2[i] + self.sym_uA[i] / p.V_i)

            # dcB/dt = flow - r1 + feed_B/V
            rhs_cB.append(flow_cB - r1[i] + self.sym_uB[i] / p.V_i)

            # dcR/dt = flow + r1 (product)
            rhs_cR.append(flow_cR + r1[i])

            # dcS/dt = flow + r2 (side product)
            rhs_cS.append(flow_cS + r2[i])

            # Temperature ODE
            # dTr/dt = flow_T + heat_transfer + reaction_heat
            heat_transfer = p.kA / (p.rho * p.cp * p.V_i) * (p.Tj - self.sym_Tr[i])
            reaction_heat = (
                Q1 * ca.exp(-p.Ea1 / (p.R_gas * T_kelvin[i])) * self.sym_cA[i] * self.sym_cB[i]
                + Q2 * ca.exp(-p.Ea2 / (p.R_gas * T_kelvin[i])) * self.sym_cA[i] ** 2
            )
            rhs_Tr.append(flow_Tr + heat_transfer + reaction_heat)

        # Stack into ODE right-hand side
        rhs = ca.vertcat(
            ca.vertcat(*rhs_cA),
            ca.vertcat(*rhs_cB),
            ca.vertcat(*rhs_cR),
            ca.vertcat(*rhs_cS),
            ca.vertcat(*rhs_Tr),
        )

        # Symbolic expression for ODE
        self.ode = rhs

        # Create CasADi function: f(x, u) -> dx/dt
        self.f_ode = ca.Function(
            "f_ode",
            [self.sym_x, self.sym_u],
            [rhs],
            ["x", "u"],
            ["xdot"],
        )

    # -------------------------------------------------------------------------
    # Integrator
    # -------------------------------------------------------------------------

    def _build_integrator(self, method: Literal["cvodes", "rk4", "explicit_euler", "implicit_euler"] = "cvodes") -> None:
        """Build a discrete-time integrator.

        Args:
            method: Integration method ('cvodes' or 'rk4').
        """
        p = self.params

        if method == "cvodes":
            # CasADi's CVODES integrator (adaptive, accurate)
            dae = {
                "x": self.sym_x,
                "p": self.sym_u,
                "ode": self.f_ode(self.sym_x, self.sym_u),
            }
            self._integrator_casadi = ca.integrator("integrator", "cvodes", dae, 0.0, p.dt)

            def integrator_fn(x: ca.SX, u: ca.SX) -> ca.SX:
                return self._integrator_casadi(x0=x, p=u)["xf"]
            
            x_next = integrator_fn(self.sym_x, self.sym_u)
        
        elif method == "explicit_euler":
            # Explicit Euler
            x_next = self.sym_x + p.dt * self.f_ode(self.sym_x, self.sym_u)

        elif method == "implicit_euler":
            # Implicit Euler: solve x_next = x + dt * f(x_next, u)
            # Use CasADi rootfinder to solve the implicit equation
            x_next_sym = ca.SX.sym("x_next", p.nx)
            x_curr_sym = ca.SX.sym("x_curr", p.nx)
            u_param_sym = ca.SX.sym("u_param", p.nu)
            
            # Residual: g(x_next; x_curr, u) = x_next - x_curr - dt * f(x_next, u) = 0
            residual = x_next_sym - x_curr_sym - p.dt * self.f_ode(x_next_sym, u_param_sym)
            
            # Create rootfinder function
            # g takes (x_next, p) where p = (x_curr, u)
            g_func = ca.Function(
                "g_implicit_euler",
                [x_next_sym, ca.vertcat(x_curr_sym, u_param_sym)],
                [residual],
                ["x", "p"],
                ["g"],
            )
            
            # Newton solver for the implicit equation
            self._rootfinder = ca.rootfinder(
                "implicit_euler_solver", "newton", g_func,
                {"error_on_fail": False}
            )
            
            # Evaluate: use current state as initial guess
            x_next = self._rootfinder(self.sym_x, ca.vertcat(self.sym_x, self.sym_u))

        elif method == "rk4":
            dt = self.params.dt
            f = self.f_ode
            k1 = f(self.sym_x, self.sym_u)
            k2 = f(self.sym_x + dt / 2 * k1, self.sym_u)
            k3 = f(self.sym_x + dt / 2 * k2, self.sym_u)
            k4 = f(self.sym_x + dt * k3, self.sym_u)
            x_next = self.sym_x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        else:
            raise ValueError(f"Unknown integrator method: {method}")

        self.f_d_sym = x_next

        self.f_d_func = ca.Function(
            "f_d",
            [self.sym_x, self.sym_u],
            [x_next],
            ["x", "u"],
            ["x_next"],
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate one time step.

        Args:
            x: Current state (nx,).
            u: Input (nu,).

        Returns:
            Next state (nx,).
        """
        x = np.atleast_1d(x).flatten()
        u = np.atleast_1d(u).flatten()
        x_next = np.array(self.f_d_func(x, u)).flatten()
        return x_next

    def simulate_trajectory(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray,
    ) -> np.ndarray:
        """Simulate a trajectory given an input sequence.

        Args:
            x0: Initial state (nx,).
            u_sequence: Input sequence (N, nu).

        Returns:
            State trajectory (N+1, nx) including initial state.
        """
        x0 = np.atleast_1d(x0).flatten()
        u_sequence = np.atleast_2d(u_sequence)

        # Handle shape: prefer (N, nu)
        if u_sequence.shape[1] != self.params.nu:
            u_sequence = u_sequence.T

        N = u_sequence.shape[0]

        # Simulate
        x_traj = np.zeros((N + 1, self.params.nx))
        x_traj[0] = x0

        for k in range(N):
            x_traj[k + 1] = self.simulate_step(x_traj[k], u_sequence[k])

        return x_traj

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def extract_reactor_state(self, x: np.ndarray, reactor_idx: int) -> dict[str, float]:
        """Extract state of a single reactor.

        Args:
            x: Full state vector (nx,).
            reactor_idx: Reactor index (0-indexed).

        Returns:
            Dictionary with state values.
        """
        x = np.atleast_1d(x).flatten()
        nr = self.params.nr
        i = reactor_idx

        return {
            "cA": x[i],
            "cB": x[nr + i],
            "cR": x[2 * nr + i],
            "cS": x[3 * nr + i],
            "Tr": x[4 * nr + i],
        }


# =============================================================================
# Input Sequence Generation
# =============================================================================


def create_step_input(
    params: CSTRParams,
    n_steps: int,
    uA_level: float = 0.5,
    uB_level: float = 0.5,
) -> np.ndarray:
    """Create a constant input sequence.

    Args:
        params: Model parameters.
        n_steps: Number of time steps.
        uA_level: Feed rate A for all reactors [mol/s].
        uB_level: Feed rate B for all reactors [mol/s].

    Returns:
        Input sequence (n_steps, nu).
    """
    nr = params.nr
    u_seq = np.zeros((n_steps, params.nu))

    u_seq[:, :nr] = uA_level  # uA
    u_seq[:, nr : 2 * nr] = uB_level  # uB

    return u_seq


def create_varying_input(
    params: CSTRParams,
    n_steps: int,
    profile: Literal["ramp", "sine", "step_changes"] = "step_changes",
) -> np.ndarray:
    """Create a time-varying input sequence.

    Args:
        params: Model parameters.
        n_steps: Number of time steps.
        profile: Type of variation.

    Returns:
        Input sequence (n_steps, nu).
    """
    nr = params.nr
    u_seq = np.zeros((n_steps, params.nu))
    t = np.arange(n_steps)

    if profile == "ramp":
        # Linear ramp for feed rates
        uA = 0.2 + 0.8 * t / n_steps
        uB = 0.4 + 0.8 * t / n_steps

    elif profile == "sine":
        # Sinusoidal variation
        phase = 2 * np.pi * t / n_steps
        uA = 0.5 + 0.3 * np.sin(phase)
        uB = 0.5 + 0.3 * np.sin(phase + np.pi / 4)

    else:  # step_changes
        # Step changes at 1/4, 1/2, 3/4 of simulation
        uA = np.zeros(n_steps)
        uB = np.zeros(n_steps)

        levels_uA = [0.3, 0.7, 0.5, 0.9]
        levels_uB = [0.3, 0.6, 0.4, 0.8]

        step_times = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4]

        for i in range(len(step_times)):
            start = step_times[i]
            end = step_times[i + 1] if i + 1 < len(step_times) else n_steps
            uA[start:end] = levels_uA[i]
            uB[start:end] = levels_uB[i]

    # Apply to all reactors
    for i in range(nr):
        u_seq[:, i] = uA
        u_seq[:, nr + i] = uB

    return u_seq


# =============================================================================
# Plotting
# =============================================================================


def plot_trajectories(
    t: np.ndarray,
    x_traj: np.ndarray,
    u_seq: np.ndarray,
    params: CSTRParams,
    title: str = "CSTR Cascade Simulation",
) -> None:
    """Plot state and input trajectories.

    Args:
        t: Time vector (N+1,).
        x_traj: State trajectory (N+1, nx).
        u_seq: Input sequence (N, nu).
        params: Model parameters.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    nr = params.nr
    colors = plt.cm.tab10(np.linspace(0, 1, max(nr, 3)))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"{title} ({nr} Reactors)", fontsize=14, fontweight="bold")

    t_u = t[:-1]  # Time for inputs

    # --- Concentration A ---
    ax = axes[0, 0]
    for i in range(nr):
        ax.plot(t, x_traj[:, i], color=colors[i], label=f"Reactor {i+1}", linewidth=1.5)
    ax.set_ylabel("$c_A$ [mol/L]")
    ax.set_title("Concentration A (Reactant)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Concentration B ---
    ax = axes[0, 1]
    for i in range(nr):
        ax.plot(t, x_traj[:, nr + i], color=colors[i], label=f"Reactor {i+1}", linewidth=1.5)
    ax.set_ylabel("$c_B$ [mol/L]")
    ax.set_title("Concentration B (Reactant)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Concentration R (Product) ---
    ax = axes[1, 0]
    for i in range(nr):
        ax.plot(t, x_traj[:, 2 * nr + i], color=colors[i], label=f"Reactor {i+1}", linewidth=1.5)
    ax.set_ylabel("$c_R$ [mol/L]")
    ax.set_title("Concentration R (Desired Product)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Concentration S (Side Product) ---
    ax = axes[1, 1]
    for i in range(nr):
        ax.plot(t, x_traj[:, 3 * nr + i], color=colors[i], label=f"Reactor {i+1}", linewidth=1.5)
    ax.set_ylabel("$c_S$ [mol/L]")
    ax.set_title("Concentration S (Side Product)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Temperature ---
    ax = axes[2, 0]
    for i in range(nr):
        ax.plot(t, x_traj[:, 4 * nr + i], color=colors[i], label=f"$T_r^{{{i+1}}}$", linewidth=1.5)
    # Tj is now fixed, show as horizontal line
    ax.axhline(params.Tj, color="gray", linestyle="--", alpha=0.6, label=f"$T_j$ (fixed)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Reactor & Jacket Temperature")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Feed Rates ---
    ax = axes[2, 1]
    for i in range(nr):
        ax.step(t_u, u_seq[:, i], color=colors[i], where="post",
                linestyle="-", label=f"$u_A^{{{i+1}}}$", linewidth=1.5)
        ax.step(t_u, u_seq[:, nr + i], color=colors[i], where="post",
                linestyle="--", label=f"$u_B^{{{i+1}}}$", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Feed rate [mol/s]")
    ax.set_title("Feed Rates")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CSTR Cascade Simulation - Simplified Model")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    nr = 5  # Number of reactors
    n_steps = 100  # Simulation steps
    dt = 1.0  # Time step [s]

    params = CSTRParams(nr=nr, dt=dt)
    print(f"\nModel: {nr} reactors, dt={dt}s")
    print(f"States: nx={params.nx} (5 per reactor)")
    print(f"Inputs: nu={params.nu} (2 per reactor: uA, uB)")
    print(f"Fixed Tj: {params.Tj}°C")
    print(f"Volume per reactor: V_i={params.V_i:.2f} L")

    # -------------------------------------------------------------------------
    # Create Models (CVODES and Implicit Euler)
    # -------------------------------------------------------------------------
    print("\nBuilding models...")
    model_cvodes = CSTRCascade(params, integrator="cvodes")
    model_implicit = CSTRCascade(params, integrator="implicit_euler")


    # -------------------------------------------------------------------------
    # Initial Conditions and Input Sequence
    # -------------------------------------------------------------------------
    x0 = params.get_default_x0()
    print(f"\nInitial state:")
    print(f"  Concentrations: {x0[:4*nr].max():.4f} mol/L (near zero)")
    print(f"  Temperature: {x0[4*nr]:.1f}°C")

    # Create time-varying input
    u_seq = create_varying_input(params, n_steps, profile="step_changes")
    print(f"\nInput sequence: step_changes profile")
    print(f"  uA range: [{u_seq[:, :nr].min():.2f}, {u_seq[:, :nr].max():.2f}] mol/s")
    print(f"  uB range: [{u_seq[:, nr:2*nr].min():.2f}, {u_seq[:, nr:2*nr].max():.2f}] mol/s")

    # -------------------------------------------------------------------------
    # Simulate with CVODES
    # -------------------------------------------------------------------------
    print("\nSimulating with CVODES...")
    x_traj_cvodes = model_cvodes.simulate_trajectory(x0, u_seq)
    print(f"  ✓ Completed {n_steps} steps")

    # -------------------------------------------------------------------------
    # Simulate with Implicit Euler
    # -------------------------------------------------------------------------
    print("Simulating with Implicit Euler...")
    x_traj_implicit = model_implicit.simulate_trajectory(x0, u_seq)
    print(f"  ✓ Completed {n_steps} steps")

    # -------------------------------------------------------------------------
    # Compare Integrators
    # -------------------------------------------------------------------------
    diff = np.abs(x_traj_cvodes - x_traj_implicit)
    print(f"\nIntegrator Comparison (CVODES vs Implicit Euler):")
    print(f"  Max absolute difference: {diff.max():.2e}")
    print(f"  Mean absolute difference: {diff.mean():.2e}")

    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\nFinal State Summary (CVODES):")
    for i in range(nr):
        state = model_cvodes.extract_reactor_state(x_traj_cvodes[-1], i)
        print(f"  Reactor {i+1}: cA={state['cA']:.3f}, cB={state['cB']:.3f}, "
              f"cR={state['cR']:.3f}, cS={state['cS']:.4f}, Tr={state['Tr']:.1f}°C")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    t = np.arange(n_steps + 1) * dt

    # Plot CVODES results
    fig1 = plot_trajectories(t, x_traj_cvodes, u_seq, params, title="CVODES Integration")
    
    # Plot Implicit Euler results
    fig2 = plot_trajectories(t, x_traj_implicit, u_seq, params, title="Implicit Euler Integration")

    # Plot comparison (difference between CVODES and Implicit Euler)
    fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig3.suptitle("Integrator Comparison: |CVODES - Implicit Euler|", fontsize=14, fontweight="bold")

    state_names = ["$c_A$", "$c_B$", "$c_R$", "$c_S$", "$T_r$"]
    for j, name in enumerate(state_names):
        ax = axes.flat[j]
        for i in range(nr):
            idx = j * nr + i
            ax.plot(t, diff[:, idx], label=f"Reactor {i+1}", linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"|Δ{name}|")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

    # Hide unused subplot
    axes.flat[-1].axis("off")
    plt.tight_layout()

    print("\n" + "=" * 60)
    print("Simulation complete! Showing plots...")
    print("=" * 60)

    plt.show()

