import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import seaborn as sns
# from scipy.stats import ortho_group, linregress
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from time import time
import json

from orthogonality_deep_representations import Data, MLP, COLORS


class DatasetCIFAR10(Dataset):

    def __init__(
            self, 
            batch_size: int, 
            drop_last: bool = False, 
            shuffle: bool = True, 
            data_dir: str = None
        ):
        """
        ARGUMENTS:
            - batch_size: number of samples per batch
            - drop_last: if the last batch is smaller than batch_size, drop it
            - shuffle: shuffle the dataset at each epoch
            - data_dir: path to store the downloaded dataset 
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dir = './data/' if data_dir is None else data_dir

        self.train_dataloader = DataLoader(
            torchvision.datasets.CIFAR10(
                root=self.data_dir, 
                train="train", 
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ])
            ),
            batch_size=self.batch_size, 
            drop_last=drop_last,
            shuffle=self.shuffle,
        )
    
    def print_infos(self):
        print(f"### Dataset CIFAR10 train split")
        print(f"- Dataset shape: {self.train_dataloader.dataset.data.shape}")
        print(f"- Number of samples: {len(self.train_dataloader.dataset)}")
        print(f"- Number of batches: {len(self.train_dataloader)}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Number of classes: {len(self.train_dataloader.dataset.classes)}")
        print(f"- Shuffle: {self.shuffle}")
        print(f"- Data dir: {self.data_dir}")
        print("")
    
    def __len__(self):
        return len(self.train_dataloader.dataset)


def save_training_data(
        epochs_data: dict[int, list[Data]], 
        save_dir: str = None,
        filename: str = None,
        **kwargs,
    ):
    """
    Save the training data produced during execution of train_MLP_CIFAR10 in a json file.
    """

    save_dir = "./outputs/" if save_dir is None else save_dir
    filename = "training_data" if filename is None else filename

    json_data = {"orthogonality_gap": {}} # {epoch_idx: [gap_batch_1, gap_batch_2, ..., gap_batch_n]}

    for key, value in kwargs.items():
        json_data[key] = value

    for epoch_idx in epochs_data:
        json_data["orthogonality_gap"][epoch_idx] = [
            data.to_dict()["orthogonality_gap"][0] for data in epochs_data[epoch_idx]
        ]

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_dir, f"{filename}.json")
    with open(file_path, "w") as f:
        f.write(json.dumps(json_data, indent=4))
        print(f"\nTraining data saved in '{file_path}'")


def train_MLP_CIFAR10(device: str = None):
    """
    Train a ReLU MLP on CIFAR10 dataset and record the loss and orthogonality gap at each epoch in a json file.
    """

    hidden_dim = 800 # 800
    depth = 20 # 20
    batch_size = 500 # 500
    epochs = 50 # 50
    step_size = 0.01 # 0.01
    device = "cpu" if device is None else device
    compute_orth_gap = True

    dataset = DatasetCIFAR10(batch_size=batch_size, drop_last=True, shuffle=True)
    dataset.print_infos()

    mlp = MLP(
        d=hidden_dim, 
        l=depth,
        in_dim=32*32*3, # 32x32 RGB images (32*32*3=3072)
        out_dim=10,
        bn=False, 
        bias=False, 
        act="ReLU",
        init_method="xavier",
        device=device,
    )
    mlp.print_infos()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=step_size)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9) # lr=lr*<gamma> each <step_size> epochs

    losses_steps = []
    losses_epochs = []
    epochs_data = defaultdict(list)
    for epoch in range(epochs):
        
        # train the network
        start_time = time()
        for inputs, labels in dataset.train_dataloader:

            optimizer.zero_grad()
            
            outputs, _ = mlp(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            
            loss.backward()
            optimizer.step()

            losses_steps.append(loss.cpu().item())
        
        print(f"Epoch {epoch+1}/{epochs} | Loss = {loss.item():.4f} [{time()-start_time:.2f}s]")

        # scheduler.step() # reduce the learning rate
        
        # compute orthogonality gap and loss over the whole dataset
        if compute_orth_gap:
            with torch.no_grad():
                losses_epochs.append(0)
                for inputs, labels in tqdm(dataset.train_dataloader, desc=f"Compute orthogonality gap and loss"):
                    outputs, batch_data = mlp(inputs.to(device), return_orth_gap=True, select_layers="last")
                    epochs_data[epoch].append(batch_data)
                    losses_epochs[-1] += batch_size * criterion(outputs, labels.to(device)).cpu().item()
                losses_epochs[-1] /= len(dataset)
    
    # plot loss at each step during optimization and save figure
    plt.plot(losses_steps, label="loss steps")
    plt.title(f"Optimization (epochs={epochs} | batch_size={batch_size})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig("./outputs/loss_steps.png")
    plt.close()

    # plot loss on the whole dataset at each epoch and save figure
    plt.plot([k for k in range(epochs)], losses_epochs, color="red", marker="o", linewidth=1, label="loss epochs")
    plt.title(f"Optimization (epochs={epochs} | batch_size={batch_size})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig("./outputs/loss_epochs.png")
    plt.close()

    # save training data in a json file
    save_training_data(
        epochs_data=epochs_data, 
        hidden_dim=hidden_dim,
        depth=depth,
        batch_size=batch_size,
        epochs=epochs,
        step_size=step_size,
        losses_steps=losses_steps,
        losses_epochs=losses_epochs,
    )


def plot_figure_4b(filepath: str = None, filename:str = None, save_dir: str = None):
    """
    ARGUMENTS:
        - filepath: name of the json file containing the training data (loss + orthogonality_gap at each epoch)
        - filename: name of the figure to save
        - save_dir: directory to save the figure
    """

    # load data
    filepath = filepath if filepath is not None else "outputs/training_data.json"
    with open(filepath, "r") as f:
        data = json.load(f)

    # define data to plot
    epochs = data["epochs"]
    # batch_size = data["batch_size"]
    hidden_dim = data["hidden_dim"]
    depth = data["depth"]
    # step_size = data["step_size"]
    orth_gap_per_epoch = [np.mean(data["orthogonality_gap"][str(i)]) for i in range(epochs)]
    losses_epochs = data["losses_epochs"]
    epochs_idx = [i for i in range(epochs)]

    # plot figure
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(epochs_idx, orth_gap_per_epoch, color=COLORS[0], marker="o", linewidth=1, label="orthogonality gap")
    ax_left.set_ylabel("orthogonality gap")
    ax_right.plot(epochs_idx, losses_epochs, color=COLORS[1], marker="o", linewidth=1, label="loss")
    ax_right.set_ylabel("loss")
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax_left.set_title(f"Gap and loss during training MLP[depth={depth},width={hidden_dim}]")
    ax_left.set_xlabel("epochs")
    fig.tight_layout()

    # save figure
    save_dir = save_dir if save_dir is not None else "outputs/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure_4b.png" if filename is None else filename))
    plt.close()


if __name__ == '__main__':

    train_MLP_CIFAR10(device="mps")

    plot_figure_4b()
