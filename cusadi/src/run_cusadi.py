import torch
import casadi as ca

from src.apply_cusadi import apply_cusadi

# build casadi function (example)
x = ca.SX.sym("x", 2)
p = ca.SX.sym("p", 2)
y = ca.sin(x) + p
fn = ca.Function("my_fn", [x, p], [y])

f_gpu = apply_cusadi(fn, batch_size=8192)

x_t = torch.randn(8192, 2, device="cuda", dtype=torch.float64)
p_t = torch.randn(8192, 2, device="cuda", dtype=torch.float64)

y_t = f_gpu(x_t, p_t)  # returns torch tensor on GPU
