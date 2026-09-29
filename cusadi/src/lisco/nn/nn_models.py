import copy
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedforwardNNConfig:
    """Hyperparameter config for FeedforwardNN. Defaults match FeedforwardNN.__init__.

    Note: n_in and n_out are excluded — they are always derived from NLP
    dimensions at runtime and are not static hyperparameters.
    """

    n_hidden_layers: int = 2
    n_neurons: int = 500
    act_fn: str = "relu"
    output_act_fn: str = "linear"
    nn_dtype: torch.dtype = torch.float32


def get_activation_layer(act_fn):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    elif act_fn == 'tanh':
        return torch.nn.Tanh()
    elif act_fn == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act_fn == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act_fn == "gelu":
        return torch.nn.GELU()
    else:
        raise ValueError("Activation function not implemented.")


class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network model.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons=500, act_fn='relu', output_act_fn='linear',nn_dtype=torch.float32):
        super().__init__()
        if n_hidden_layers <= 0:
            raise ValueError("Number of hidden layers must be > 0.")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers+1
        self.n_neurons = n_neurons
        self.nn_dtype = nn_dtype
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.layers = torch.nn.ModuleList()

        # add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))

        # add output layer
        self.layers.append(torch.nn.Linear(n_neurons, n_out,dtype=self.nn_dtype))
        if output_act_fn != 'linear':
            self.layers.append(get_activation_layer(output_act_fn))

    def forward(self, x):
        x = x.to(self.nn_dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def parametrize_output(self):
        # Find the last linear layer (output layer) and apply weight normalization only to it.
        for i in range(len(self.layers) - 1, -1, -1):  # Iterate backwards
            layer = self.layers[i]
            if isinstance(layer, torch.nn.Linear):
                # Apply weight normalization to the output layer
                self.layers[i] = torch.nn.utils.parametrizations.weight_norm(layer)
                break  # Stop after parametrizing the first linear layer found (from the end)

    def remove_parametrizations(self):
        # Function to remove the parametrization of all layers if these layers are parameterized
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear) and torch.nn.utils.parametrize.is_parametrized(layer, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(layer, 'weight', leave_parametrized=True)

    def to_dict(self) -> dict:
        """Serialize architecture hyperparameters to a JSON-safe plain dict."""
        return {
            "n_in":            self.n_in,
            "n_out":           self.n_out,
            "n_hidden_layers": self.n_hidden_layers,
            "n_neurons":       self.n_neurons,
            "act_fn":          self.act_fn,
            "output_act_fn":   self.output_act_fn,
            "nn_dtype":        str(self.nn_dtype).split(".")[-1],
        }

    def save_weights(self, save_dir, file_name="weights.pt"):
        save_pth = Path(save_dir, file_name)
        torch.save(self.state_dict(), save_pth)

    def load_weights(self, load_dir, file_name="weights.pt", map_location=None):
        load_pth = Path(load_dir, file_name)
        if map_location is None:
            map_location = next(self.parameters()).device
        self.load_state_dict(torch.load(load_pth, map_location=map_location, weights_only=True))




@torch.no_grad()
def count_params(net):
    n_params = sum([param.numel() for param in net.parameters()])
    return n_params


def export_jit_cpu(model):
    mod_cpu = copy.deepcopy(model).cpu()
    mod_cpu = torch.jit.script(mod_cpu)
    mod_cpu = torch.jit.optimize_for_inference(mod_cpu)
    return mod_cpu


def export_jit_gpu(model):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. export_jit_gpu requires a CUDA-capable runtime.")
    mod_gpu = copy.deepcopy(model).cuda()
    mod_gpu = torch.jit.script(mod_gpu)
    mod_gpu = torch.jit.optimize_for_inference(mod_gpu)
    return mod_gpu
