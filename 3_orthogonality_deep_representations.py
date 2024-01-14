import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import ortho_group
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


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

    def forward(self, x, bn: bool = None, return_similarity: bool = False, return_orth_gap: bool = False) -> torch.Tensor | dict:
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
            data = defaultdict(list)
            data["type"] = "BN" if bn else "Vanilla"
            data["layer"] = [0]
            data["cosine_similarity"].append(self.__cosine_similarity(x)) if return_similarity else None
            data["orthogonality_gap"].append(self.__orthogonality_gap(x)) if return_orth_gap else None


        for layer_idx in range(self.l):

            x = self.linear_layers[layer_idx](x)
    
            if self.activation is not None:
                x = self.activations[layer_idx](x)

            if bn:
                x = self.bn_layers[layer_idx](x) / np.sqrt(self.d)
        
            if return_metrics:
                data["layer"].append(layer_idx+1)
                data["cosine_similarity"].append(self.__cosine_similarity(x)) if return_similarity else None
                data["orthogonality_gap"].append(self.__orthogonality_gap(x)) if return_orth_gap else None
        
        if return_metrics:
            return data
        else:
            return x


class Data():
    """
    Data structure to store data from multiple forward passes of the MLP with cosine similarity recorded.
    """

    def __init__(self, d: int = None, l: int = None, save_dir: str = None):
        self.d = d
        self.l = l
        self.layers = []
        self.cos_sim = []
        self.orth_gaps = []
        self.types = []
        self.save_dir = save_dir if save_dir is not None else "outputs/"

    def add_run(self, data: defaultdict):
        """ Add data from a single forward pass to the data structure. """
        for layer in data["layer"]:
            self.layers.append(layer)
            self.types.append(data["type"])
        for cos_sim in data["cosine_similarity"]:
            self.cos_sim.append(cos_sim)
        for orth_gap in data["orthogonality_gap"]:
            self.orth_gaps.append(orth_gap)
    
    def get_data(self) -> dict:
        """ Return stored data as a dict. """
        data_dict = {
            "layer": self.layers,
            "type": self.types,
        }
        if len(self.cos_sim) > 0:
            data_dict["cosine_similarity"] = self.cos_sim
        if len(self.orth_gaps) > 0:
            data_dict["orthogonality_gap"] = self.orth_gaps
        if self.d is not None:
            data_dict["d"] = [self.d for _ in range(len(self.layers))]
        return data_dict
    
    def plot_similarity_accross_layers(self, save_dir: str = None):

        data_dict = self.get_data()

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
        # plot.set_xticks(np.arange(0, self.l, 1))
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
    
    def plot_orth_gap_accross_layers(self, clusters: str = None, data_dict: dict = None, save_dir: str = None, filename: str = None):
        """
        ARGUMENTS:
            clusters: how to group the data in the plot: "type" for Vanilla vs. BN or "d" for width comparison
            data_dict: if not None, then plot the data in data_dict instead of the stored data
            save_dir: directory where to save the plot
            filename: name of the file to save the plot
        """

        data_dict = self.get_data() if data_dict is None else data_dict

        if len(data_dict["orthogonality_gap"]) == 0:
            raise ValueError("No orthogonality gap data stored...")
        
        clusters = clusters if clusters is not None else "type"
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
        # plot.set_xticks(np.arange(0, self.l, 1))
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

    data = Data(d=d, l=l)

    for _ in tqdm(range(n_runs), desc=f"Forward pass for network width={d}"):

        mlp = MLP(d=d, l=l, activation=None)

        # Vanilla
        xVanilla = create_orth_vectors(d=d, n=2)
        data.add_run(mlp(xVanilla, return_similarity=True, return_orth_gap=True, bn=False))

        # BN
        xBN = create_correlated_vectors(d=d, n=2, eps=eps)
        data.add_run(mlp(xBN, return_similarity=True, return_orth_gap=True, bn=True))

    data.plot_similarity_accross_layers()
    data.plot_orth_gap_accross_layers(clusters="type", filename="figure_2a_Vanilla_vs_BN.png")


def plots_figure_2a(n_runs: int = 20):
    """
    Figure 2a: Orthogonality gap vs. depth
    """

    d = 32
    l = 50
    batch_size = 4

    data_cluster = {}

    for d in [32, 512]:

        data_cluster[d] = Data(d=d, l=l)

        for _ in tqdm(range(n_runs), desc=f"Forward pass for network width={d}"):

            mlp = MLP(d=d, l=l, activation=None)

            xBN = create_correlated_vectors(d=d, n=batch_size, eps=0.001)
            data_cluster[d].add_run(mlp(xBN, return_orth_gap=True, bn=True))


    merged_data = defaultdict(list)
    for d, data in data_cluster.items():
        data_dict = data.get_data()
        for key in data_dict:
            merged_data[key].extend(data_dict[key])
    
    Data().plot_orth_gap_accross_layers(clusters="d", data_dict=merged_data)


if __name__ == "__main__":

    plots_figure_1()
    plots_figure_2a()

