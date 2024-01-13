import os
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
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
            # lin_layer.weight.data = torch.randn(self.d, self.d) / np.sqrt(self.d)

        self.activations = [self.ACTIVATIONS[self.activation]() for _ in range(l)] if self.activation is not None else None
        
        self.bn_layers = [BN(self.d) for _ in range(self.l)]

        self.data = defaultdict(list)

    def forward(self, x, bn: bool = None, return_similarity: bool = False) -> torch.Tensor | dict:
        """
        ARGUMENTS:
            x: input tensor of shape (batch_size, d) is equal to H.T defined in the article, be careful...
            bn: batch normalization
            return_similarity: if True then record and return the cosine similarity between pairs of sample with column indexes (2k,2k+1)
        """

        if bn is None:
            bn = self.bn
        
        if return_similarity:
            if x.shape[0] != 2:
                raise ValueError(f"If return_similarity is True then input must contain exactly 2 samples but got {x.shape[0]}")
            data = defaultdict(list)
            data["type"] = "BN" if bn else "Vanilla"
            data["layer"] = [0]
            data["cosine_similarity"].append(abs(torch.cosine_similarity(x[:,0], x[:,1], dim=0).item()))

        for layer_idx in range(self.l):

            x = self.linear_layers[layer_idx](x)
    
            if self.activation is not None:
                x = self.activations[layer_idx](x)

            if bn:
                x = self.bn_layers[layer_idx](x) / np.sqrt(self.d)
        
            if return_similarity:
                data["layer"].append(layer_idx+1)
                data["cosine_similarity"].append(abs(torch.cosine_similarity(x[0,:], x[1,:], dim=0).item()))
        
        if return_similarity:
            return data
        else:
            return x


class Data():
    """
    Data structure to store data from multiple forward passes of the MLP with cosine similarity recorded.
    """

    def __init__(self, l: int = None, save_dir: str = None):
        self.l = l
        self.layers = []
        self.cos_sim = []
        self.types = []
        self.save_dir = save_dir if save_dir is not None else "outputs/"

    def add_run(self, data: dict):
        """ Add data from a single forward pass to the data structure. """
        for layer, cos_sim in zip(data["layer"], data["cosine_similarity"]):
            self.layers.append(layer)
            self.cos_sim.append(cos_sim)
            self.types.append(data["type"])
    
    def get_data(self) -> dict:
        """ Return stored data as a dict. """
        return {
            "layer": self.layers,
            "cosine_similarity": self.cos_sim,
            "type": self.types,
        }
    
    def plot_similarity_accross_layers(self, save_dir: str = None):

        if len(self.cos_sim) == 0:
            raise ValueError("No data stored...")
        
        # create save directory if it does not exist
        save_dir = save_dir if save_dir is not None else self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_dir, "figure_1.png")
        
        # create plot
        plot = sns.lineplot(
            data=self.get_data(), 
            x="layer",
            y="cosine_similarity",
            hue="type",
            marker="o",
            errorbar=('ci', 95),
        )
        # plot.set_xticks(np.arange(0, self.l, 1))
        plot.set_xlabel("layer", fontsize=10)
        plot.set_ylabel("cosine similarity",fontsize=10)
        plot.set_title("Cosine similarity between pairs of samples accross layers", fontsize=12)
        handles, labels = plot.get_legend_handles_labels()
        plot.legend(handles=handles, labels=labels)

        # save plot
        plot.get_figure().savefig(filename, format="png", dpi=300)
        

if __name__ == "__main__":

    d = 32
    l = 50
    n_runs = 20
    eps = 0.001

    data = Data()

    for run_id in tqdm(range(n_runs)):

        mlp = MLP(d=d, l=l, activation=None)

        # Vanilla
        ## TODO : Generate random pairs of orthogonal vectors
        # xVanilla = torch.randn(2, d)
        xVanilla = torch.zeros((2, d)) # orthogonal initialization
        xVanilla[0,0] = 1
        xVanilla[1,1] = 1
        data.add_run(mlp(xVanilla, return_similarity=True, bn=False))

        # BN
        ## TODO : Generate random pairs of vectors with high cosine similarity
        # xBN = torch.randn(2, d)
        xBN = torch.zeros((2, d)) # dependent initialization
        xBN[0,0] = 1
        xBN[1,0] = 1-eps
        xBN[1,1] = eps
        data.add_run(mlp(xBN, return_similarity=True, bn=True))

    data.plot_similarity_accross_layers()


