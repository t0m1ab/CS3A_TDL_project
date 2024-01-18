import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import ortho_group, linregress
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

COLORS = sns.color_palette()


class Data():
    """
    Data structure to store data from multiple forward pass of a single MLP with fixed length and width.
    """

    def __init__(self, d: int = None, l: int = None, save_dir: str = None):
        self.d = d
        self.l = l
        self.layers = []
        self.types = []
        self.cos_sim = []
        self.orth_gaps = []
        self.orth_radius = 0
        self.save_dir = save_dir if save_dir is not None else "outputs/"

    def add_values(self, network_type: str, layer_idx: int, cos_sim: float = None, orth_gap: float = None):
        """ Add data from a single forward pass to the data structure. """
        self.types.append(network_type)
        self.layers.append(layer_idx)
        if cos_sim is not None:
            self.cos_sim.append(cos_sim)
        if orth_gap is not None:
            self.orth_gaps.append(orth_gap)
    
    def to_dict(self) -> dict:
        """ Return stored data as a dict. """
        data_dict = {
            "layer": self.layers,
            "type": self.types,
            "d": [self.d for _ in range(len(self.layers))], 
        }
        if len(self.cos_sim) > 0:
            data_dict["cosine_similarity"] = self.cos_sim
        if len(self.orth_gaps) > 0:
            data_dict["orthogonality_gap"] = self.orth_gaps
        return data_dict
    
    def plot_similarity_accross_layers(self, data_dict: dict = None, save_dir: str = None):
        """
        ARGUMENTS:
            data_dict: if not None, then plot the data in data_dict instead of the stored data
        """

        data_dict = self.to_dict() if data_dict is None else data_dict

        if len(data_dict["cosine_similarity"]) == 0:
            raise ValueError("No cosine similarity data stored...")
        
        # create plot
        plot = sns.lineplot(
            data=data_dict, 
            x="layer",
            y="cosine_similarity",
            hue="type",
            marker="o",
            errorbar=('ci', 95),
            palette=sns.color_palette(n_colors=len(set(data_dict["type"]))),
        )
        plot.set_xlabel("layer", fontsize=10)
        plot.set_ylabel("cosine similarity",fontsize=10)
        plot.set_title(f"Cosine similarity between pair of samples accross layers (d={self.d})", fontsize=12)
        handles, labels = plot.get_legend_handles_labels()
        plot.legend(handles=handles, labels=labels)

        # create save directory if it does not exist and save plot
        save_dir = save_dir if save_dir is not None else self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_dir, "figure_1.png")
        plot.get_figure().savefig(filename, format="png", dpi=300)
        plt.close()
    
    def plot_orth_gap_accross_layers(
            self, 
            clusters: str = None, 
            data_dict: dict = None, 
            save_dir: str = None, 
            filename: str = None,
        ):
        """
        ARGUMENTS:
            clusters: how to group the data in the plot: "type" for Vanilla vs. BN or "d" for width comparison
            data_dict: if not None, then plot the data in data_dict instead of the stored data
            save_dir: directory where to save the plot
            filename: name of the file to save the plot
        """

        data_dict = self.to_dict() if data_dict is None else data_dict

        if len(data_dict["orthogonality_gap"]) == 0:
            raise ValueError("No orthogonality gap data stored...")
        
        clusters = clusters if clusters is not None else "type" # define clusters for grouping values in the plot
        data_dict["log_orthogonality_gap"] = [np.log(orth_gap) for orth_gap in data_dict["orthogonality_gap"]]
        
        # create plot
        plt.figure()
        plot = sns.lineplot(
            data=data_dict, 
            x="layer",
            y="log_orthogonality_gap",
            hue=clusters,
            marker="o",
            errorbar=('ci', 95),
            palette=sns.color_palette(n_colors=len(set(data_dict[clusters]))),
        )
        plot.set_xlabel("layer", fontsize=10)
        plot.set_ylabel("log(orthogonality gap)",fontsize=10)
        plot_title = f"Orthogonality gap accross layers (d={self.d})" if self.d is not None else "Orthogonality gap accross layers"
        plot.set_title(plot_title, fontsize=12)
        handles, labels = plot.get_legend_handles_labels()
        plot.legend(handles=handles, labels=[f"d={d}" for d in labels] if clusters != "type" else labels)

        # create save directory if it does not exist and save plot
        save_dir = save_dir if save_dir is not None else self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_dir, "figure_2a.png" if filename is None else filename)
        plot.get_figure().savefig(filename, format="png", dpi=300)
        plt.close()   

    def plot_orth_radius_vs_width(
            self, 
            data_dict: dict = None, 
            save_dir: str = None, 
            filename: str = None,
        ):
        """
        ARGUMENTS:
            data_dict: if not None, then plot the data in data_dict instead of the stored data
            save_dir: directory where to save the plot
            filename: name of the file to save the plot
        """

        data_dict = self.to_dict() if data_dict is None else data_dict

        if len(data_dict["log_avg_orthogonality_gap"]) == 0:
            raise ValueError("No avg orthogonality gap data stored...")

        slope_mean = np.mean(data_dict["slope"])
        slope_std = np.std(data_dict["slope"])
        data_dict.pop("slope")
                
        # create plot
        plt.figure()
        plot = sns.lineplot(
            data=data_dict, 
            x="log_d",
            y="log_avg_orthogonality_gap",
            marker="o",
            errorbar=('ci', 95),
        )
        plot.set_xlabel("log(width)", fontsize=10)
        plot.set_ylabel("log(avg orthogonality gap)",fontsize=10)
        plot.set_title(f"Avg orthogonality gap vs. network width (slope={slope_mean:.2f}+-{slope_std:.2f})", fontsize=12)

        # create save directory if it does not exist and save plot
        save_dir = save_dir if save_dir is not None else self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_dir, "figure_2b.png" if filename is None else filename)
        plot.get_figure().savefig(filename, format="png", dpi=300)
        plt.close() 


class BN(nn.Module):
    """ 
    Custom Batch Normalization layer as defined in the article.
    """

    def __init__(self, input_dim: int, device: str = None):
        super().__init__()
        self.input_dim = input_dim
        self.device = device if device is not None else "cpu"

    def forward(self, x, eps: float = 1e-6):
        """ x is equal to M.T defined in the article, be careful... """
        u = torch.diag(x.t() @ x).reshape(-1, 1) + eps # add 1e-6 to avoid future possible division by 0
        diag = torch.eye(self.input_dim, device=self.device) * torch.pow(u, -0.5)
        return x @ diag.t()


class SinActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class PaperSigmoidActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1 + self.tanh(x)


class MLP(nn.Module):

    LAYERS_SELECTION = ["all", "first", "last", "first_last"]

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "paper_sigmoid": PaperSigmoidActivation,
        "tanh": nn.Tanh,
        "sin": SinActivation,
    }

    INIT_METHODS = ["xavier", "normal", "orthogonal"]

    def __init__(
            self, 
            d: int, 
            l: int, 
            in_dim: int = None, 
            out_dim: int = None, 
            bn: bool = False, 
            bias: bool = False, 
            act: str = None,
            init_method: str = None,
            init_batch: torch.Tensor = None,
            device: str = None,
        ):
        """ 
        ARGUMENTS:
            - d: input dimension = hidden dimension = output dimension
            - l: number layers
            - in_dim: input dimension of the network (set to d if None)
            - out_dim: output dimension of the network (set to d if None)
            - bn: batch normalization
            - bias: if true then add bias to linear layers
            - act: act function in [None, "relu", "sigmoid", "tanh"]
            - init_method: weight initialization method in MLP.INIT_METHODS
            - init_batch: batch of vectors to use for weight initialization if method is "orthogonal"
            - device: device to use for computations in ["cpu", "mps", "cuda"]
        """
        super(MLP, self).__init__()

        if d < 1:
            raise ValueError(f"d={d} must be striclty positive")
        if l < 1:
            raise ValueError(f"d={l} must be striclty positive")
        self.d = d
        self.l = l
        self.input_dim = in_dim if in_dim is not None else d
        self.output_dim = out_dim if out_dim is not None else d
        self.bn = bn
        self.bias = bias
        self.act = None
        if (act is not None) and (act.lower() in MLP.ACTIVATIONS):
            self.act = act.lower()
        device = "cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not built with MPS enabled.")
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
        self.device = torch.device(device)

        # define linear layers: input_dim -> hidden_dim -> ...[l times]... -> hidden_dim -> output_dim
        self.flatten = nn.Flatten()
        self.input_layer = None
        if self.input_dim != self.d:
            self.input_layer = nn.Linear(self.input_dim, self.d, bias=self.bias, device=self.device)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.d, self.d, bias=self.bias, device=self.device) for _ in range(self.l)])
        self.output_layer = None
        if self.output_dim != self.d:
            self.output_layer = nn.Linear(self.d, self.output_dim, bias=self.bias, device=self.device)

        # only applied after hidden layers
        self.activations = nn.ModuleList([self.ACTIVATIONS[self.act]() for _ in range(l)]) if self.act is not None else None
        
        # only applied after hidden layers
        self.bn_layers = nn.ModuleList([BN(self.d, device=self.device) for _ in range(self.l)])

        # init layers weights
        self.init_method = self.__init_layers(init_method, init_batch)
    
    def print_infos(self):
        print(f"### MLP[d={self.d}, l={self.l}, bn={self.bn}, bias={self.bias}, act={self.act}, init={self.init_method}] ###")
        for idx, (name, layer) in enumerate(self.named_parameters()):
            print(f"{idx} - {name} | shape = {tuple(layer.size())}")
        print("")
    
    def __init_layer(self, weight: nn.Module, method: str):
        if method == "normal":
            torch.nn.init.normal_(weight, mean=0, std=1.0/np.sqrt(self.d))
        elif method == "xavier":
            torch.nn.init.xavier_uniform_(weight, gain=torch.nn.init.calculate_gain("relu"))
        else:
            print(f"WARNING: init method '{method}' not recognized, no specific initialization was performed...")

    def __init_layers(self, init_method, init_batch):
        """ Initliaze the layers weights according to init_method. """

        init_method = init_method if init_method in MLP.INIT_METHODS else "xavier"
        
        if init_method != "orthogonal": # no iterative process required
            if self.input_layer is not None:
                self.__init_layer(self.input_layer.weight, method=init_method)
            if self.output_layer is not None:
                self.__init_layer(self.output_layer.weight, method=init_method)
            for layer in self.hidden_layers:
                self.__init_layer(layer.weight, method=init_method)
        
        else: # iterative orthogonalization process

            if init_batch is None:
                raise ValueError("init_batch must be provided when init_method is 'orthogonal'")

            # init input and output layers using xavier
            if self.input_layer is not None:
                self.__init_layer(self.input_layer.weight, method="xavier")
            if self.output_layer is not None:
                self.__init_layer(self.output_layer.weight, method="xavier")
            
            x = self.flatten(init_batch)
            
            if self.input_layer is not None:
                x = self.input_layer(x)

            for layer_idx in range(self.l):

                # init the layer using the incoming representations of the batch
                with torch.no_grad():
                    v, sigma, ut = torch.svd(x) # not implemented for MPS as of January 17th 2024 so will run on CPU
                    left = v[:self.d,:self.d]
                    inverse_sqrt_sigma = torch.diag(torch.pow(sigma[:self.d], -0.5))
                    right = ut[:self.d,:self.d]
                    orth_weights = left.mm(inverse_sqrt_sigma).mm(right)
                    self.hidden_layers[layer_idx].weight.data = orth_weights
                    norm_factor = torch.norm(self.hidden_layers[layer_idx](x))
                    self.hidden_layers[layer_idx].weight.data = orth_weights / norm_factor

                # forward through the layer (+ activation + batch norm if required)
                x = self.hidden_layers[layer_idx](x)
                x = self.activations[layer_idx](x) if self.act is not None else x
                x = self.bn_layers[layer_idx](x) / np.sqrt(self.d) if self.bn else x
            
            if self.output_layer is not None:
                x = self.output_layer(x)

        return init_method

    def __cosine_similarity(self, x: torch.Tensor) -> float:
        """
        ARGUMENTS:
            x: input tensor of shape (batch_size, d) is equal to H.T defined in the article, be careful...
        """
        if x.shape[0] > 2:
            print("WARNING: Cosine similarity is only computed between the first 2 samples of the batch.")
        if x.shape[0] < 2:
            raise ValueError(f"At least 2 samples are required to compute the cosine similarity and got {x.shape[0]}")
        return abs(torch.cosine_similarity(x[0,:], x[1,:], dim=0).item())
    
    def __orthogonality_gap(self, x: torch.Tensor) -> float:
        """
        ARGUMENTS:
            x: input tensor of shape (batch_size, d) is equal to H.T defined in the article, be careful...
        """
        batch_size = x.shape[0]
        gap = (x @ x.t())/(torch.norm(x).item()**2) - torch.eye(batch_size, device=self.device)/batch_size
        return torch.norm(gap).item()

    def __layer_check(self, layer_idx: int, l: int, select_layers: str):
        """ Check if layer_idx is of select_layers type according to the depth of the network l. """
        if select_layers == "all":
            return True
        elif select_layers == "first":
            return layer_idx == 0
        elif select_layers == "last":
            return layer_idx == l
        elif select_layers == "first_last":
            return (layer_idx == 0) or (layer_idx == l)
        else:
            raise ValueError(f"select_layers must be in {self.LAYERS_SELECTION} but is {select_layers}")

    def forward(
            self, 
            x: torch.Tensor, 
            bn: bool = None, 
            return_similarity: bool = False, 
            return_orth_gap: bool = False,
            select_layers: str = None,
        ) -> tuple[torch.Tensor, Data]:
        """
        ARGUMENTS:
            x: input tensor of shape (batch_size, d) is equal to H.T defined in the article, be careful...
            bn: batch normalization
            return_similarity: if True then record and return the cosine similarity between the first pair of sample in the batch x at each layer matching select_layers
            return_orth_gap: if True then record and return the orthogonality gap at each layer matching select_layers
            select_layers: one element of MLP.LAYERS_SELECTION
        """

        if bn is None:
            bn = self.bn

        return_data = return_similarity or return_orth_gap
        select_layers = select_layers if select_layers is not None else "all"
        if not select_layers in self.LAYERS_SELECTION:
            raise ValueError(f"select_layers must be in {self.LAYERS_SELECTION} but is {select_layers}")

        data = Data(d=self.d, l=self.l)
        
        # flatten input (do nothing if input is already a batch of vectors (2D tensor))
        x = self.flatten(x)
        
        # input layer
        if self.input_layer is not None:
            x = self.input_layer(x)

        if return_data and self.__layer_check(0, self.l, select_layers):
            data.add_values(
                network_type="BN" if bn else "Vanilla", 
                layer_idx=0, 
                cos_sim=self.__cosine_similarity(x) if return_similarity else None,
                orth_gap=self.__orthogonality_gap(x) if return_orth_gap else None,
            )

        # hidden layers
        for layer_idx in range(self.l):

            x = self.hidden_layers[layer_idx](x)
    
            if self.act is not None:
                x = self.activations[layer_idx](x)

            if bn:
                x = self.bn_layers[layer_idx](x) / np.sqrt(self.d)
        
            if return_data and self.__layer_check(layer_idx+1, self.l, select_layers):
                data.add_values(
                    network_type="BN" if bn else "Vanilla", 
                    layer_idx=layer_idx+1, 
                    cos_sim=self.__cosine_similarity(x) if return_similarity else None,
                    orth_gap=self.__orthogonality_gap(x) if return_orth_gap else None,
                )
        
        # output layer
        if self.output_layer is not None:
            x = self.output_layer(x)
        
        return x, data


def create_orth_vectors(d: int, n: int, device: str = None) -> torch.Tensor:
    """
    Create and return n orthogonal vectors of dimension d.
    ARGUMENTS:
        d: dimension of the vectors
        n: number of vectors to return
    """
    if n > d:
        raise ValueError(f"n={n} must be smaller than d={d}")
        
    orth_mat = ortho_group.rvs(dim=d)
    rand_indexes = np.random.choice(d, size=n, replace=False)

    orth_vectors = torch.tensor(orth_mat[rand_indexes, :], dtype=torch.float32, device=device)

    return orth_vectors.to("cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device)


def create_correlated_vectors(d: int, n: int, eps: float = 0.001, device: str = None) -> torch.Tensor:
    """
    Create and return n correlated vectors of dimension d.
    ARGUMENTS:
        d: dimension of the vectors
        n: number of vectors to return
        eps: level of correlation between the vectors
    """
    if n > d:
        raise ValueError(f"n={n} must be smaller than d={d}")
        
    x = torch.zeros((n, d))
    x[:,0] = 1
    for i in range(1, n):
        x[i,i] = eps
    
    return x.to("cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device)


def plots_figure_1(n_runs: int = 20, device: str = None):
    """
    Figure 1: Orthogonality: BN vs. Vanilla networks
    """

    d = 32
    l = 50
    eps = 0.001

    device = "cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device

    data_cluster = []

    for _ in tqdm(range(n_runs), desc=f"Cosine similarity stats over {n_runs} runs"):

        mlp = MLP(d=d, l=l, act=None, device=device)

        # Vanilla
        xVanilla = create_orth_vectors(d=d, n=2, device=device)
        data_cluster.append(mlp(xVanilla, return_similarity=True, return_orth_gap=True, bn=False)[1])

        # BN
        xBN = create_correlated_vectors(d=d, n=2, eps=eps, device=device)
        data_cluster.append(mlp(xBN, return_similarity=True, return_orth_gap=True, bn=True)[1])
    
    # merge data from different runs
    merged_data = defaultdict(list)
    for data in data_cluster:
        for key, values in data.to_dict().items():
            merged_data[key].extend(values)
    
    # use the plot methods of Data class to plot the results
    Data(d=d, l=l).plot_similarity_accross_layers(data_dict=merged_data)
    # Data(d=d, l=l).plot_orth_gap_accross_layers(clusters="type", data_dict=merged_data, filename="figure_2a_Vanilla_vs_BN.png")


def plots_figure_2a(n_runs: int = 20, device: str = None):
    """
    Figure 2a: Orthogonality gap vs. depth
    """

    l = 50
    batch_size = 4

    device = "cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device

    data_cluster = []

    for _ in tqdm(range(n_runs), desc=f"Orthogonality gap vs. network depth stats over {n_runs} runs"):
        
        for d in [32, 512]:

            mlp = MLP(d=d, l=l, act=None, device=device)

            xBN = create_correlated_vectors(d=d, n=batch_size, eps=0.001, device=device)
            data_cluster.append(mlp(xBN, return_orth_gap=True, bn=True)[1])

    # merge data from different runs
    merged_data = defaultdict(list)
    for data in data_cluster:
        for key, values in data.to_dict().items():
            merged_data[key].extend(values)
    
    # use the plot methods of Data class to plot the results
    Data().plot_orth_gap_accross_layers(clusters="d", data_dict=merged_data)


def plots_figure_2b(n_runs: int = 20, act: str = None, device: str = None):
    """
    Figure 2b: Orthogonality gap vs. width
    """

    l = 500
    batch_size = 4

    device = "cpu" if (device is None) or (not device in ["cpu", "mps", "cuda"]) else device

    merged_data = defaultdict(list)

    for _ in tqdm(range(n_runs), desc=f"Orthogonality gap vs. network width stats over {n_runs} runs (act={act})"):

        data_dicts = []

        for d in [16, 32, 64, 128, 256, 512]:

            mlp = MLP(d=d, l=l, act=act, device=device)

            xBN = torch.randn((batch_size, d), device=device)
            data_dicts.append(mlp(xBN, return_orth_gap=True, bn=True)[1].to_dict())
        
        # compute log average of orthogonality gap accross layers and the slope wrt log width
        log_avg_orth_gap = [np.log(np.mean(data["orthogonality_gap"][1:])) for data in data_dicts]
        log_d = [np.log(data["d"][0]) for data in data_dicts]
        slope = linregress(x=log_d, y=log_avg_orth_gap).slope

        merged_data["log_avg_orthogonality_gap"].extend(log_avg_orth_gap)
        merged_data["log_d"].extend(log_d)
        merged_data["slope"].append(slope)
    
    # use the plot methods of Data class to plot the results
    filename = f"figure_2b_{act}.png" if act is not None else "figure_2b.png"
    Data().plot_orth_radius_vs_width(data_dict=merged_data, filename=filename)


if __name__ == "__main__":

    DEVICE = "cpu"

    plots_figure_1(device=DEVICE)

    plots_figure_2a(device=DEVICE)

    plots_figure_2b(device=DEVICE)

    plots_figure_2b(device=DEVICE, act="relu")
    plots_figure_2b(device=DEVICE, act="tanh")
    plots_figure_2b(device=DEVICE, act="sigmoid")
    plots_figure_2b(device=DEVICE, act="sin")