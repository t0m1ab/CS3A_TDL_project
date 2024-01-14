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


class Data():
    """
    Data structure to store data from multiple forward passes of the MLP with cosine similarity recorded.
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


class BN(nn.Module):
    """ 
    Custom Batch Normalization layer as defined in the article.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        """ x is equal to M.T defined in the article, be careful... """
        diag = torch.eye(self.input_dim) * (1 / torch.sqrt(torch.diag(x.t() @ x).reshape(-1, 1)))
        return x @ diag.t()


class MLP(nn.Module):

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    def __init__(self, d: int, l: int, bn: bool = False, bias: bool = False, activation: str = None):
        """ 
        ARGUMENTS:
            d: input dimension = hidden dimension = output dimension
            l: number layers
            bn: batch normalization
            activation: activation function in [None,"relu","sigmoid","tanh"]
        """
        super(MLP, self).__init__()

        self.d = d
        self.l = l
        self.bn = bn
        self.bias = bias
        self.activation = None
        if activation is not None:
            activation = activation.lower()
            self.activation = activation if activation in self.ACTIVATIONS else None

        self.linear_layers = [nn.Linear(self.d, self.d, bias=self.bias) for _ in range(self.l)]
        for lin_layer in self.linear_layers: # init with i.i.d gaussian weights 
            torch.nn.init.normal_(lin_layer.weight, mean=0, std=1.0/np.sqrt(self.d))

        self.activations = [self.ACTIVATIONS[self.activation]() for _ in range(l)] if self.activation is not None else None
        
        self.bn_layers = [BN(self.d) for _ in range(self.l)]

        self.data = defaultdict(list)
    
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
        gap = (x @ x.t())/(torch.norm(x).item()**2) - torch.eye(batch_size)/batch_size
        return torch.norm(gap).item()

    def forward(
            self, 
            x: torch.Tensor, 
            bn: bool = None, 
            return_similarity: bool = False, 
            return_orth_gap: bool = False,
        ) -> torch.Tensor | Data:
        """
        ARGUMENTS:
            x: input tensor of shape (batch_size, d) is equal to H.T defined in the article, be careful...
            bn: batch normalization
            return_similarity: if True then record and return the cosine similarity between the first pair of sample in the batch x at each layer
            return_orth_gap: if True then record and return the orthogonality gap at each layer
        """

        if bn is None:
            bn = self.bn
        
        return_metrics = return_similarity or return_orth_gap
        
        if return_metrics:
            data = Data(d=self.d, l=self.l)
            data.add_values(
                network_type="BN" if bn else "Vanilla", 
                layer_idx=0, 
                cos_sim=self.__cosine_similarity(x) if return_similarity else None,
                orth_gap=self.__orthogonality_gap(x) if return_orth_gap else None,
            )

        for layer_idx in range(self.l):

            x = self.linear_layers[layer_idx](x)
    
            if self.activation is not None:
                x = self.activations[layer_idx](x)

            if bn:
                x = self.bn_layers[layer_idx](x) / np.sqrt(self.d)
        
            if return_metrics:
                data.add_values(
                    network_type="BN" if bn else "Vanilla", 
                    layer_idx=layer_idx+1, 
                    cos_sim=self.__cosine_similarity(x) if return_similarity else None,
                    orth_gap=self.__orthogonality_gap(x) if return_orth_gap else None,
                )
        
        if return_metrics:
            return data
        else:
            return x


def create_orth_vectors(d: int, n: int) -> torch.Tensor:
    """
    ARGUMENTS:
        d: dimension of the vectors
        n: number of vectors to return
    RETURNS:
        torch.Tensor of shape (n, d) containing n orthogonal vectors of dimension d
    """
    if n > d:
        raise ValueError(f"n={n} must be smaller than d={d}")
    
    orth_mat = ortho_group.rvs(dim=d)
    rand_indexes = np.random.choice(d, size=n, replace=False)

    return torch.tensor(orth_mat[rand_indexes, :], dtype=torch.float32)


def create_correlated_vectors(d: int, n: int, eps: float = 0.001) -> torch.Tensor:
    """
    ARGUMENTS:
        d: dimension of the vectors
        n: number of vectors to return
        eps: level of correlation between the vectors
    RETURNS:
        torch.Tensor of shape (n, d) containing n correlated vectors of dimension d
    """
    if n > d:
        raise ValueError(f"n={n} must be smaller than d={d}")
    
    x = torch.zeros((n, d))
    x[:,0] = 1
    for i in range(1, n):
        x[i,i] = eps
    return x


def plots_figure_1(n_runs: int = 20):
    """
    Figure 1: Orthogonality: BN vs. Vanilla networks
    """

    d = 32
    l = 50
    eps = 0.001

    data_cluster = []

    for _ in tqdm(range(n_runs), desc=f"Cosine similarity stats over {n_runs} runs"):

        mlp = MLP(d=d, l=l, activation=None)

        # Vanilla
        xVanilla = create_orth_vectors(d=d, n=2)
        data_cluster.append(mlp(xVanilla, return_similarity=True, return_orth_gap=True, bn=False))

        # BN
        xBN = create_correlated_vectors(d=d, n=2, eps=eps)
        data_cluster.append(mlp(xBN, return_similarity=True, return_orth_gap=True, bn=True))
    
    merged_data = defaultdict(list)
    for data in data_cluster:
        for key, values in data.to_dict().items():
            merged_data[key].extend(values)
    
    Data(d=d, l=l).plot_similarity_accross_layers(data_dict=merged_data)
    # Data(d=d, l=l).plot_orth_gap_accross_layers(clusters="type", data_dict=merged_data, filename="figure_2a_Vanilla_vs_BN.png")


def plots_figure_2a(n_runs: int = 20):
    """
    Figure 2a: Orthogonality gap vs. depth
    """

    l = 50
    batch_size = 4

    data_cluster = []

    for _ in tqdm(range(n_runs), desc=f"Orthogonality gap vs. network depth stats over {n_runs} runs"):
        
        for d in [32, 512]:

            mlp = MLP(d=d, l=l, activation=None)

            xBN = create_correlated_vectors(d=d, n=batch_size, eps=0.001)
            data_cluster.append(mlp(xBN, return_orth_gap=True, bn=True))

    merged_data = defaultdict(list)
    for data in data_cluster:
        for key, values in data.to_dict().items():
            merged_data[key].extend(values)
    
    Data().plot_orth_gap_accross_layers(clusters="d", data_dict=merged_data)


def plots_figure_2b(n_runs: int = 20):
    """
    Figure 2b: Orthogonality gap vs. width
    """

    l = 500
    batch_size = 4

    merged_data = defaultdict(list)

    for _ in tqdm(range(n_runs), desc=f"Orthogonality gap vs. network width stats over {n_runs} runs"):

        data_dicts = []

        for d in [16, 32, 64, 128, 256, 512]:

            mlp = MLP(d=d, l=l, activation=None)

            xBN = create_correlated_vectors(d=d, n=batch_size, eps=0.001)
            data_dicts.append(mlp(xBN, return_orth_gap=True, bn=True).to_dict())
        
        log_avg_orth_gap = [np.log(np.mean(data["orthogonality_gap"][1:])) for data in data_dicts]
        log_d = [np.log(data["d"][0]) for data in data_dicts]
        slope = linregress(x=log_d, y=log_avg_orth_gap).slope

        merged_data["log_avg_orthogonality_gap"].extend(log_avg_orth_gap)
        merged_data["log_d"].extend(log_d)
        merged_data["slope"].append(slope)
    
    Data().plot_orth_radius_vs_width(data_dict=merged_data)


if __name__ == "__main__":

    plots_figure_1()
    plots_figure_2a()
    plots_figure_2b()