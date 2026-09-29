import numpy as np

lisco_path = (
    "results/case_study/"
    "linearized_four_tank_bonmin_20260916_205859.npz"
)
bonmin_path = (
    "results/case_study/"
    "linearized_four_tank_bonmin_20260913_204128.npz"
)

with np.load(lisco_path) as lisco, np.load(bonmin_path) as bonmin:
    z_lisco = lisco["z_pred"].reshape((-1,10,40))#["binary_inputs"]
    z_bonmin = bonmin["z_pred"].reshape((-1,10,40))#["binary_inputs"]
    T_lisco=lisco["Tk_val"]
    success_lisco = lisco["success"]
    success_bonmin = bonmin["success"]

    x0_lisco = lisco["initial_states"]
    x0_bonmin = bonmin["initial_states"]
    if not np.allclose(x0_lisco, x0_bonmin):
        raise ValueError(
            "The LISCO and Bonmin files do not contain the same initial states."
        )

    if z_lisco.shape != z_bonmin.shape:
        raise ValueError(
            f"Incompatible shapes: {z_lisco.shape} and {z_bonmin.shape}"
        )
    valid = (
            success_lisco
            & success_bonmin
            & np.isfinite(z_lisco).all(axis=-1)
            & np.isfinite(z_bonmin).all(axis=-1)
    )

    # Protect against numerical output such as 0.999999999.
    z_lisco_integer = np.rint(z_lisco).astype(int)
    z_bonmin_integer = np.rint(z_bonmin).astype(int)

    # Whether each individual binary variable agrees.
    variable_matches = z_lisco_integer == z_bonmin_integer

    # Whether both binary variables agree at a closed-loop step.
    step_matches = variable_matches.all(axis=-1)

    n_valid_steps = int(valid.sum())
    n_exact_steps = int((step_matches & valid).sum())

    n_valid_variables = int(valid.sum() * z_lisco.shape[-1])
    n_correct_variables = int(
        (variable_matches & valid[..., None]).sum()
    )



    print(
        f"Correct individual binary variables: "
        f"{n_correct_variables}/{n_valid_variables} "
        f"({n_correct_variables / n_valid_variables:.1%})"
    )