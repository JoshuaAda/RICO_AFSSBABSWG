import scienceplots
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

matplotlib.use('TkAgg')
plt.style.use('science')

sys.path.append('./Figures')
import config_mpl

lisco_path = (
    "results/case_study/"
    "linearized_four_tank_lisco_20260915_162332.npz"#"linearized_four_tank_lisco_20260913_151740.npz"
)
armijo_path = (
    "results/case_study/"
    "linearized_four_tank_lisco_20260916_112225.npz"
)

#with np.load(lisco_path) as lisco, np.load(armijo_path) as armijo:
lisco=np.load(lisco_path)
armijo=np.load(armijo_path)
T_lisco=lisco["Tk_val"]
T_armijo=armijo["Tk_val"]
fig, axes = plt.subplots(1,1,figsize=(config_mpl.textwidth, config_mpl.textwidth / config_mpl.golden_ratio),sharex="col")
axes.hist(np.log10(T_lisco),bins=100,color="blue",alpha=0.7,label="LISCO")
axes.hist(np.log10(T_armijo),bins=100,color="orange",alpha=0.7,label="Armijo")
axes.axvline(x=-5, color="black", linestyle="--",label=r"$T_{\text{lim}}$")
axes.legend()
fig.tight_layout()
plt.xlabel(r"$\log{T_{m_{\text{final}}}}$ [-]")
plt.ylabel("Frequency [-]")

#plt.show()
plt.savefig("convergence.pdf")
