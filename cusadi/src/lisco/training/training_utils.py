"""Training utilities: logger, optimizer, gradient clipping, and finalization."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch


class TrainLogger:
    def __init__(self, buffer_size=100, device=torch.device('cuda')):
        """
        Initialize TrainLogger with GPU buffering capability.

        Args:
            buffer_size (int): Number of entries to accumulate in GPU buffer before flushing to CPU
            device (str): Device to store buffers on ('cuda' or 'cpu')
        """
        self.history = {}
        self.buffer_size = buffer_size
        self.device = device if torch.cuda.is_available() else torch.device('cpu')

        # Buffers: dict of {key: tensor}
        self.buffers = {}
        self.buffer_counts = {}  # Track how many entries in each buffer

    def _init_buffer(self, key):
        if key not in self.buffers:
            dev = self.device
            self.buffers[key] = torch.empty(self.buffer_size, device=dev, dtype=torch.float32)
            self.buffer_counts[key] = 0

    def _flush_buffer(self, key):
        """Flush buffer contents to CPU history."""
        if key not in self.buffers or self.buffer_counts[key] == 0:
            return

        # Get valid entries from buffer
        count = self.buffer_counts[key]
        buffer_data = self.buffers[key][:count].detach().cpu().tolist()

        # Add to history
        if key not in self.history:
            self.history[key] = []
        self.history[key].extend(buffer_data)

        # Reset buffer count
        self.buffer_counts[key] = 0

    def flush_all_buffers(self):
        """Flush all buffers to CPU history."""
        for key in list(self.buffers.keys()):
            self._flush_buffer(key)

    def log_value_buffered(self, val, key):
        # normalize to float32 tensor on target device
        if torch.is_tensor(val):
            t = val.detach()
            if t.device != self.device:
                t = t.to(self.device, non_blocking=True)
            # always cast to float32
            if t.dtype != torch.float32:
                t = t.float()
        else:
            # python numbers become float32
            t = torch.tensor(val, device=self.device, dtype=torch.float32)

        # initialize buffer if needed
        if key not in self.buffers:
            self._init_buffer(key)

        # ensure scalar
        if t.numel() != 1:
            raise ValueError(f"log_value_buffered('{key}'): expected scalar, got shape {tuple(t.shape)}")

        idx = self.buffer_counts[key]
        self.buffers[key][idx] = t
        self.buffer_counts[key] += 1

        if self.buffer_counts[key] >= self.buffer_size:
            self._flush_buffer(key)

    def log_values_buffered(self, kv: dict):
        # e.g., {"loss": loss_scalar_tensor, "acc": acc_scalar_tensor}
        for k, v in kv.items():
            self.log_value_buffered(v, k)

    def log_value(self, val, key):
        """Original immediate logging (for backward compatibility)."""
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        assert isinstance(val, (int, float)), "Value must be a scalar."
        if key not in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def log_data(self, val, key):
        """Log data that overwrites previous value."""
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        self.history[key] = val

    def log_array(self, val, key):
        """Log array/tensor as list."""
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        val = val.tolist()
        self.history[key] = val

    def calculate_metrics(self, val: torch.Tensor):
        assert torch.is_tensor(val), "Value must be a tensor."
        # promote to float for quantiles if needed
        if not val.is_floating_point():
            val = val.float()
        qs = torch.tensor([0.99, 0.95, 0.50, 0.01], device=val.device)
        q_vals = torch.quantile(val, qs)  # shape [4]
        val_mean = val.mean()
        val_max  = val.max()
        val_min  = val.min()
        # unpack
        val_99, val_95, val_50, val_1 = q_vals.unbind()
        return val_mean, val_max, val_min, val_99, val_95, val_50, val_1

    def log_metrics(self, val, key, use_buffer=False):
        # WARNING: This function leads to non-negligible overhead if used frequently during training!
        """
        Log distribution metrics for a tensor.

        Args:
            val: Input tensor
            key: Base key for metrics
            use_buffer: If True, use buffered logging
        """
        val_mean, val_max, val_min, val_99, val_95, val_50, val_1 = self.calculate_metrics(val)

        log_fn = self.log_value_buffered if use_buffer else self.log_value

        log_fn(val_mean, key + "_mean")
        log_fn(val_max, key + "_max")
        log_fn(val_min, key + "_min")
        log_fn(val_99, key + "_99")
        log_fn(val_95, key + "_95")
        log_fn(val_50, key + "_50")
        log_fn(val_1, key + "_1")

    def save_history(self, exp_pth, file_name="history", as_json=True):
        """Save history to disk. Flushes all buffers first."""
        # Ensure all buffers are flushed before saving
        self.flush_all_buffers()

        exp_pth = Path(exp_pth)
        exp_pth.mkdir(parents=True, exist_ok=True)

        # Save as .pt
        torch.save(self.history, exp_pth.joinpath(file_name + ".pt"))

        # Save as JSON
        if as_json:
            with open(exp_pth.joinpath(file_name + ".json"), 'w') as f:
                json.dump(self.history, f, indent=4)

        print("History saved to: ", exp_pth.joinpath(file_name))

    def plot_history(
        self,
        keys_to_plot,
        save_path=None,
        x_key="epoch",
        title=None,
        yscale=None,
    ):
        """Plot one or multiple history keys and optionally save figure.

        Args:
            keys_to_plot: List of keys (single axis) or list of lists (one subplot per inner list).
            save_path: Optional path to save the figure (.png, etc.).
            x_key: History key for x-axis. Falls back to index if unavailable/mismatched.
            title: Optional figure title.
            yscale: Optional y-scale ("log") applied to all subplots.
        """
        self.flush_all_buffers()

        if not keys_to_plot:
            raise ValueError("keys_to_plot must not be empty.")

        if isinstance(keys_to_plot[0], str):
            groups = [list(keys_to_plot)]
        else:
            groups = [list(group) for group in keys_to_plot]

        n_axes = len(groups)
        fig, axes = plt.subplots(n_axes, 1, figsize=(9, 3.5 * n_axes), squeeze=False)
        axes = axes[:, 0]

        for ax, group in zip(axes, groups):
            plotted_any = False
            for key in group:
                if key not in self.history:
                    continue

                y_vals = self.history[key]
                x_vals = self.history.get(x_key)
                if isinstance(x_vals, list) and len(x_vals) == len(y_vals):
                    ax.plot(x_vals, y_vals, label=key)
                else:
                    ax.plot(y_vals, label=key)
                plotted_any = True

            ax.set_xlabel(x_key if x_key in self.history else "index")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            if yscale is not None:
                ax.set_yscale(yscale)
            if plotted_any:
                if len(group) == 1:
                    ax.set_title(group[0])
                else:
                    ax.set_title(", ".join(group))
                ax.legend()
            else:
                ax.set_title("No matching history keys")

        if title is not None:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
        else:
            fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

        return fig, axes

    def visualize_history(self, key, log_scaling=False, exp_pth=None, save_fig=False):
        """Visualize history for a key."""
        # Flush buffers to ensure all data is available
        self.flush_all_buffers()

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        if log_scaling:
            ax.set_yscale('log')

        if key in self.history:
            ax.plot(self.history[key], label=key)
        else:
            print(f"Key '{key}' not found in history.")

        ax.legend()
        plt.show()
        if save_fig:
            assert exp_pth is not None, "exp_pth must be provided."
            plt.savefig(Path(exp_pth).joinpath(key + ".png"))
        return fig, ax

    def post_calculate_ema_resample_metrics(self, batch_size, ema_beta=0.95):
        """Calculate EMA-based resampling metrics and convergence fractions."""
        # Flush buffers to ensure all data is available
        self.flush_all_buffers()

        n_resample_values = self.history.get('n_resample', [])
        n_resample_tol_values = self.history.get('n_resample_tol', [])
        n_resample_iter_values = self.history.get('n_resample_iter', [])

        n_resample_ema_values = []
        n_resample_tol_ema_values = []
        n_resample_iter_ema_values = []
        conv_frac_samples_values = []
        conv_frac_batch_values = []

        n_resample_ema = 1.0
        n_resample_tol_ema = 0.0
        n_resample_iter_ema = 0.0

        for n_resample, n_resample_tol, n_resample_iter in zip(n_resample_values, n_resample_tol_values, n_resample_iter_values):
            # Update EMAs
            n_resample_ema = ema_beta * n_resample_ema + (1 - ema_beta) * n_resample
            n_resample_tol_ema = ema_beta * n_resample_tol_ema + (1 - ema_beta) * n_resample_tol
            n_resample_iter_ema = ema_beta * n_resample_iter_ema + (1 - ema_beta) * n_resample_iter

            # Calculate convergence fractions
            conv_frac_samples = n_resample_tol_ema / (n_resample_ema + 1e-16)
            conv_frac_batch = n_resample_tol_ema / batch_size

            # Store values
            n_resample_ema_values.append(n_resample_ema)
            n_resample_tol_ema_values.append(n_resample_tol_ema)
            n_resample_iter_ema_values.append(n_resample_iter_ema)
            conv_frac_samples_values.append(conv_frac_samples)
            conv_frac_batch_values.append(conv_frac_batch)

        # Update history
        self.history['n_resample_ema'] = n_resample_ema_values
        self.history['n_resample_tol_ema'] = n_resample_tol_ema_values
        self.history['n_resample_iter_ema'] = n_resample_iter_ema_values
        self.history['conv_frac_samples'] = conv_frac_samples_values
        self.history['conv_frac_batch'] = conv_frac_batch_values


def create_optimizer(params, lr, weight_decay, use_amsgrad=True):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=use_amsgrad)


def adaptive_clip_grad(params, clip_factor=0.01, eps=1e-3):
    """Per-parameter adaptive gradient clipping (AGC, Brock et al. 2021)."""
    for p in params:
        if p.grad is None:
            continue
        param_norm = torch.linalg.vector_norm(p.detach(), ord=2)
        grad_norm = torch.linalg.vector_norm(p.grad.detach(), ord=2)
        max_norm = (param_norm + eps) * clip_factor
        if grad_norm > max_norm:
            p.grad.mul_(max_norm / (grad_norm + 1e-16))


def finalize_training(train_logger, train_tic, epoch, log_values=True, **extra_log_data):
    """Compute train time, log final data, print summary."""
    train_toc = time.perf_counter()
    train_time = train_toc - train_tic

    if log_values:
        train_logger.log_data(train_time, "train_time")
        for key, value in extra_log_data.items():
            if isinstance(value, torch.Tensor):
                train_logger.log_array(value.detach().cpu(), key)
            else:
                train_logger.log_data(value, key)
        train_logger.flush_all_buffers()

    print("#----------------------------------#")
    print("Training complete.")
    print(f"Total number of epochs: {epoch + 1}")
    print(f"Total training time: {train_time:.2f} sec")
    print("#----------------------------------#")

    return train_logger


def generate_experiment_dir(log_dirs=["runs"], save_dir=None):
    for i in range(1000):
        local_pth = Path(*log_dirs, f"exp_{i}")
        if not local_pth.exists():
            break
        if i == 999:
            raise ValueError("Experiment directory could not be generated.")
    if save_dir is None:
        exp_pth = Path(__file__).parent.absolute()
    else:
        exp_pth = Path(save_dir)
    exp_pth = exp_pth.joinpath(local_pth)
    exp_pth.mkdir(parents=True, exist_ok=True)
    return exp_pth
