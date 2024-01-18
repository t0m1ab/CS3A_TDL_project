import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from time import time
import json

from orthogonality_deep_representations import MLP, COLORS


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
    
    def sample_random_batch(self, batch_size: int = None):
        """ Sample a random batch of size batch_size from the dataset. """

        if batch_size is None: # sample a batch of size self.batch_size
            for batch, _ in self.train_dataloader:
                return batch

        c = 0
        samples = []
        for batch, _ in self.train_dataloader:
            samples.append(batch)
            c += batch.shape[0]
            if c >= batch_size:
                break
        
        return torch.cat(samples, dim=0)[:batch_size]
    
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


def plot_losses(
        batch_size: int, 
        epochs: int, 
        losses_steps: list[float], 
        losses_epochs: list[float],
        save_dir: str = None,
    ):
    """
    Plot the following curves: loss at each step during optimization and loss on the whole dataset at each epoch.
    ARGUMENTS:
        - batch_size: number of samples per batch
        - epochs: number of epochs
        - losses_steps: list of losses on the whole dataset at each epoch
        - losses_epochs: list of losses at each step during optimization
        - save_dir: directory to save the figures
    """

    save_dir = "outputs/" if save_dir is None else save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # plot loss at each step during optimization and save figure
    plt.plot(losses_steps, label="loss steps")
    plt.title(f"Optimization (epochs={epochs} | batch_size={batch_size})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    filename = os.path.join(save_dir, "loss_steps.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nLoss at each step during optimization saved in '{filename}'")

    # plot loss on the whole dataset at each epoch and save figure
    plt.plot([k for k in range(epochs)], losses_epochs, color="red", marker="o", linewidth=1, label="loss epochs")
    plt.title(f"Optimization (epochs={epochs} | batch_size={batch_size})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    filename = os.path.join(save_dir, "loss_epochs.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nLoss at each step during optimization saved in '{filename}'")


def save_training_data(
        epochs_data: dict[int, list[float]], 
        save_dir: str = None,
        **kwargs,
    ):
    """
    Save the training data produced during execution of train_MLP_CIFAR10 in a json file.
    ARGUMENTS:
        - epochs_data: dict containing the orthogonality gap for each batch at each epoch
        - save_dir: directory to save the json file
        - kwargs: additional data to save in the json file (like depth, width, batch_size, loss...)
    """

    # prepare data to save in json
    json_data = {"orthogonality_gap": epochs_data}
    for key, value in kwargs.items():
        json_data[key] = value

    # save data in json file
    save_dir = "./outputs/" if save_dir is None else save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_dir, "training_data.json")
    with open(file_path, "w") as f:
        f.write(json.dumps(json_data, indent=4))
        print(f"\nTraining data saved in '{file_path}'")


def train_MLP_CIFAR10(
        hidden_dim: int = 800,
        depth: int = 20,
        batch_size: int = 500,
        epochs: int = 50,
        step_size: float = 0.01,
        init_method: str = None,
        compute_orth_gap: bool = True,
        device: str = None,
        plot_figures: bool = True,
        save_data: bool = True,
        save_dir: str = None,
    ):
    """
    Train a ReLU MLP on CIFAR10 dataset and record the loss and orthogonality gap at each epoch in a json file.
    ARGUMENTS:
        - hidden_dim: number of neurons per hidden layer
        - depth: number of hidden layers
        - batch_size: number of samples per batch
        - epochs: number of epochs
        - step_size: learning rate
        - init_method: initialization method for the weights must be in MLP.INIT_METHODS
        - compute_orth_gap: if True, compute the orthogonality gap before training and after each epoch on the whole dataset
        - device: device to use for training in ["cpu", "cuda", "mps"]
        - plot_figures: if True, plot the loss curves
        - save_data: if True, save the training data in a json file
        - save_dir: directory to save loss plots and training data
    """

    device = "cpu" if device is None else device

    dataset = DatasetCIFAR10(batch_size=batch_size, drop_last=True, shuffle=True)
    dataset.print_infos()

    init_method = "xavier" if init_method is None else init_method
    init_batch = None
    if init_method == "orthogonal": # sample n=3d samples from the dataset as initialization batch
        init_batch = dataset.sample_random_batch(batch_size=3*hidden_dim).to(device)

    mlp = MLP(
        d=hidden_dim, 
        l=depth,
        in_dim=32*32*3, # 32x32 RGB images (32*32*3=3072)
        out_dim=10,
        bn=False, 
        bias=False, 
        act="ReLU",
        init_method=init_method,
        init_batch=init_batch,
        device=device,
    )
    mlp.print_infos()

    epochs_data = defaultdict(list) # store the orthogonality gap for all batches at each epoch

    # compute orthogonality gap before training over the whole dataset
    if compute_orth_gap:
        with torch.no_grad():
            for inputs, labels in tqdm(dataset.train_dataloader, desc=f"Compute orthogonality gap at initialization"):
                _, batch_data = mlp(inputs.to(device), return_orth_gap=True, select_layers="last")
                epochs_data[0].append(batch_data.orth_gaps[-1]) # only save the orthogonality gap at the last layer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=step_size)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9) # lr=lr*<gamma> each <step_size> epochs

    losses_steps = [] # store the loss at each optimization step
    losses_epochs = [] # store the loss on the whole dataset at each epoch
    print(f"\n### Start training for {epochs} epochs")
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
        
        # scheduler.step() # reduce the learning rate
        
        # compute loss and orthogonality gap over the whole dataset
        with torch.no_grad():
            losses_epochs.append(0)
            tqdm_desc = "Compute loss and orthogonality gap" if compute_orth_gap else "Compute loss"
            for inputs, labels in tqdm(dataset.train_dataloader, desc=tqdm_desc):
                outputs, batch_data = mlp(inputs.to(device), return_orth_gap=compute_orth_gap, select_layers="last")
                losses_epochs[-1] += batch_size * criterion(outputs, labels.to(device)).cpu().item()
                if compute_orth_gap: # only save the orthogonality gap at the last layer (that's why [-1])
                    epochs_data[epoch+1].append(batch_data.orth_gaps[-1])
            losses_epochs[-1] /= len(dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss = {losses_epochs[-1]:.4f} [{time()-start_time:.2f}s]")
    
    if plot_figures: # plot losses
        plot_losses(
            batch_size=batch_size, 
            epochs=epochs, 
            losses_steps=losses_steps, 
            losses_epochs=losses_epochs,
            save_dir=save_dir,
        )

    if save_data: # save training data in a json file
        save_training_data(
            epochs_data=epochs_data,
            hidden_dim=hidden_dim,
            depth=depth,
            batch_size=batch_size,
            epochs=epochs,
            step_size=step_size,
            losses_steps=losses_steps,
            losses_epochs=losses_epochs,
            save_dir=save_dir,
        )


def plot_figure_3a(directories: list[str] = None, save_dir: str = None):
    """
    Figure 3a: gap and loss vs. depth
    ARGUMENTS:
        - directories: list of directories where to find the training data json file for each MLP
        - save_dir: directory to save the figure
    """

    # load data
    data_list = []
    for dirname in directories:
        filepath = os.path.join(dirname, "training_data.json")
        if not os.path.exists(filepath):
            raise ValueError(f"File '{filepath}' not found")
        with open(filepath, "r") as f:
            data_list.append(json.load(f))

    # define data to plot
    epochs = None
    depths = []
    end_loss_per_depth = []
    orth_gap_per_depth = []
    for data in data_list:
        if epochs is not None and epochs != data["epochs"]:
            raise ValueError(f"Epochs mismatch for different trainings: {epochs} != {data['epochs']}")
        epochs = data["epochs"]
        depths.append(data["depth"])
        end_loss_per_depth.append(data["losses_epochs"][-1])
        if not "0" in data["orthogonality_gap"]:
            raise ValueError(f"Orthogonality gap data at initialization not found for training with depth {depths[-1]}")
        orth_gap_per_depth.append(np.mean(data["orthogonality_gap"]["0"]))

    # plot figure
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(depths, orth_gap_per_depth, color=COLORS[0], marker="o", linewidth=1, label="orthogonality gap")
    ax_left.set_ylabel("orthogonality gap")
    ax_right.plot(depths, end_loss_per_depth, color=COLORS[1], marker="o", linewidth=1, label="loss")
    ax_right.set_ylabel(f"training loss after {epochs} epochs")
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax_left.set_title(f"Initial gap and final loss for MLP with different depths")
    ax_left.set_xlabel("depth")
    fig.tight_layout()

    # save figure
    save_dir = save_dir if save_dir is not None else "outputs/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure_3a.png"))
    plt.close()


def plot_figure_3b(filepath: str = None, save_dir: str = None):
    """
    Figure 3b: gap and loss during training
    ARGUMENTS:
        - filepath: name of the json file containing the training data (loss + orthogonality_gap at each epoch)
        - save_dir: directory to save the figure
    """

    # load data
    filepath = filepath if filepath is not None else "outputs/training_data.json"
    with open(filepath, "r") as f:
        data = json.load(f)

    # define data to plot
    epochs = data["epochs"]
    hidden_dim = data["hidden_dim"]
    depth = data["depth"]
    epochs_idx = [i for i in range(epochs)]
    losses_epochs = data["losses_epochs"]
    orth_gap_per_epoch = []
    for epoch_idx in range(1, epochs+1):
        if not str(epoch_idx) in data["orthogonality_gap"]:
            raise ValueError(f"Orthogonality gap data for epoch {epoch_idx} not found in '{filepath}'")
        orth_gap_per_epoch.append(np.mean(data["orthogonality_gap"][str(epoch_idx)]))

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
    plt.savefig(os.path.join(save_dir, "figure_3b.png"))
    plt.close()


def plot_figure_4(
        directories_1: list[str] = None, 
        directories_2: list[str] = None, 
        label_1: str = None,
        label_2: str = None,
        save_dir: str = None,
    ):
    """
    Figure 4: final loss vs. depth when using xavier or orthogonal initialization
    ARGUMENTS:
        - directories_1: list of directories where to find the training data json file for each MLP of type 1
        - directories_2: list of directories where to find the training data json file for each MLP of type 2
        - label_1: label for the MLP of type 1
        - label_2: label for the MLP of type 2
        - save_dir: directory to save the figure
    """

    if len(directories_1) != len(directories_2):
        raise ValueError(f"Number of directories for MLP of type 1 and MLP of type 2 must be the same")

    # load data from directories in directories_1 and directories_2
    data_list_1 = []
    data_list_2 = []
    for dirname1, dirname2 in zip(directories_1, directories_2):
        filepath1 = os.path.join(dirname1, "training_data.json")
        filepath2 = os.path.join(dirname2, "training_data.json")
        if not os.path.exists(filepath1):
            raise ValueError(f"File '{filepath1}' not found")
        if not os.path.exists(filepath2):
            raise ValueError(f"File '{filepath2}' not found")
        with open(filepath1, "r") as f:
            data_list_1.append(json.load(f))
        with open(filepath2, "r") as f:
            data_list_2.append(json.load(f))

    # define data to plot
    epochs = None
    depths = []
    end_loss_per_depth_1 = []
    end_loss_per_depth_2 = []
    for data1, data2 in zip(data_list_1, data_list_2):
        if epochs is not None and epochs != data1["epochs"]:
            raise ValueError(f"Epochs mismatch for different trainings: {epochs} != {data1['epochs']}")
        if epochs is not None and epochs != data2["epochs"]:
            raise ValueError(f"Epochs mismatch for different trainings: {epochs} != {data2['epochs']}")
        epochs = data1["epochs"]
        depths.append(data1["depth"])
        end_loss_per_depth_1.append(data1["losses_epochs"][-1])
        end_loss_per_depth_2.append(data2["losses_epochs"][-1])

    # plot figure
    plt.plot(depths, end_loss_per_depth_1, color=COLORS[0], marker="o", linewidth=1, label=label_1)
    plt.plot(depths, end_loss_per_depth_2, color=COLORS[1], marker="o", linewidth=1, label=label_2)
    plt.xlabel("depth")
    plt.ylabel(f"training loss after {epochs} epochs")
    plt.legend(loc="upper left")
    plt.title(f"Final loss for MLP with different depths and init methods")

    # save figure
    save_dir = save_dir if save_dir is not None else "outputs/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure_4.png"))
    plt.close()


if __name__ == '__main__':

    DEVICE = "mps"
    depths = [15, 30, 45, 60, 75]

    # # reproduce Figure 3a
    for depth in depths:
        train_MLP_CIFAR10(
            epochs=30, 
            depth=depth, 
            device=DEVICE, 
            save_dir=f"outputs/training_figure_3a_depth={depth}/"
        )
    plot_figure_3a(
        directories=[f"outputs/training_figure_3a_depth={depth}/" for depth in depths],
        save_dir="outputs/"
    )

    # # reproduce Figure 3b
    train_MLP_CIFAR10(device=DEVICE, save_dir="outputs/training_figure_3b/")
    plot_figure_3b(filepath="outputs/training_figure_3b/training_data.json")

    # reproduce Figure 4
    for depth in depths:
        train_MLP_CIFAR10(
            epochs=30, 
            depth=depth, 
            init_method="orthogonal",
            compute_orth_gap=False,
            device=DEVICE,  
            save_dir=f"outputs/training_figure_4_depth={depth}/"
        )
    plot_figure_4(
        directories_1=[f"outputs/training_figure_3a_depth={depth}/" for depth in depths],
        directories_2=[f"outputs/training_figure_4_depth={depth}/" for depth in depths],
        label_1="xavier",
        label_2="orthogonal",
        save_dir="outputs/"
    )
