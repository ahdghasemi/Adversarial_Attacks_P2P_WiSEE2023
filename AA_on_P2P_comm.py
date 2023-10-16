"""
Python Script for Adversarial Attack on P2P communications.

This script contains several functions and import statements necessary for a 
wireless network simulation. It also demonstrates installing dependencies,
importing custom modules, and using a variety of tools in the wireless 
communication context.
    
Author: Ahmad Ghasemi
Date: 10/12/2023
"""

import numpy.linalg as npl
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import scipy
import random
import sys
import os
import itertools
from collections import defaultdict
from tqdm import tqdm
from google.colab import drive
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN

# Ensure to describe the purpose of custom modules.
from FPLinQ import FP_optimize, FP  # Module handling Fixed-Point Link Quality?
import wireless_networks_generator as wg  # Module for generating wireless networks?
import helper_functions  # Module containing auxiliary functions for the main script?

# Potential Functions for Reference or Use - Detail these as per actual functionality.
import function_wmmse_powercontrol as wf
import numpy.linalg as npl


def format_pytorch_version(version):
    """
    Format PyTorch version string for compatibility with installation.

    Args:
    - version (str): Original version string from torch.__version__

    Returns:
    - str: Formatted version string
    """
    return version.split('+')[0]


def format_cuda_version(version):
    """
    Format CUDA version string for compatibility with installation.

    Args:
    - version (str): Original version string from torch.version.cuda

    Returns:
    - str: Formatted version string
    """
    return 'cu' + version.replace('.', '')


# Automated Installation of PyTorch Geometric Dependencies
def install_pytorch_geometric_dependencies():
    """
    Install necessary libraries for PyTorch Geometric based on system’s PyTorch and CUDA versions.
    Note: This function is specifically tailored for Google Colab and might not work in other environments.
    """
    TORCH_version = format_pytorch_version(torch.__version__)
    CUDA_version = format_cuda_version(torch.version.cuda)

    !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{TORCH_version}+{CUDA_version}.html
    !pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{TORCH_version}+{CUDA_version}.html
    !pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{TORCH_version}+{CUDA_version}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH_version}+{CUDA_version}.html
    !pip install torch-geometric


# Mount Google Drive for Google Colab
def mount_google_drive():
    """
    Mount Google Drive to access files and folders in Google Colab.

    Note: This function requires user authorization.
    """
    drive.mount('/content/drive/', force_remount=True)
    sys.path.append('/content/drive/My Drive/Colab Notebooks/AttacksAndDefenseNew/DeepRobust-master/examples/graph')

def high_rank_approximation(A, k):
    """
    High Rank Approximation (HRA) using Singular Value Decomposition.
    
    Parameters:
    - A (numpy.ndarray): Input matrix to be approximated.
    - k (int): The rank for approximation.
    
    Returns:
    numpy.ndarray: Approximated matrix.
    
    Example:
    >>> A = np.array([[1,2], [3,4]])
    >>> approx_A = high_rank_approximation(A, 1)
    """
    u, ev, vh = npl.svd(A)
    ev[:k] = 0
    return u @ np.diag(ev) @ vh


def pytorch_hra(A, k):
    """
    PyTorch-based High Rank Approximation (HRA).
    
    Parameters:
    - A (torch.Tensor): Input matrix to be approximated.
    - k (int): The rank for approximation.
    
    Returns:
    torch.Tensor: Approximated matrix with an additional size-1 dimension.
    
    Example:
    >>> A = torch.tensor([[1,2], [3,4]])
    >>> approx_A = pytorch_hra(A, 1)
    """
    u, ev, vh = torch.linalg.svd(A)
    return (u @ torch.diag(ev) @ vh).unsqueeze(0)

import numpy as np

def our_remove_attack_budget_constraint(y, remove_perturbation_percentage):
    """
    Function to remove attack budget constraint.
    
    Parameters:
        y: Tensor
            Input tensor with certain specifications.
        remove_perturbation_percentage: float
            Percentage of perturbation to be removed.
    
    Returns:
        y: Tensor
            Modified input tensor.
    """
    y_temp = y.numpy()
    _, r, _ = y_temp.shape
    
    # Extracting desired and undesired pathloss gain samples and computing statistics
    desired_pathloss_gain_samples = y_temp[0].diagonal()
    desired_pathloss_gain_samples_min = desired_pathloss_gain_samples.min()
    desired_pathloss_gain_samples_max = desired_pathloss_gain_samples.max()
    loc = np.where(y_temp[0] != np.multiply(y_temp[0], np.eye(r)))
    undesired_pathloss_no_gain_samples = y_temp[0][loc]
    
    # Calculating the budget and initializing epsilon
    budget = remove_perturbation_percentage * np.sum(desired_pathloss_gain_samples)
    epsilon = desired_pathloss_gain_samples_min
    
    # Determining the maximum location
    maxloc = np.where(y_temp[0] == desired_pathloss_gain_samples_max)
    
    # Logic to modify the tensor based on conditions
    if desired_pathloss_gain_samples_max > budget:
        y_temp[0][maxloc] = desired_pathloss_gain_samples_max - budget
        budget = -1
    else:
        nodecounter = r
        while budget >= 0 and nodecounter >= 0:
            desired_pathloss_gain_samples_max = desired_pathloss_gain_samples.max()
            maxloc = np.where(y_temp[0] == desired_pathloss_gain_samples.max())
            maxloc = tuple(map(tuple, maxloc))
            if budget - (desired_pathloss_gain_samples_max - epsilon) >= 0:
                y_temp[0][maxloc] = epsilon
                budget = budget - (desired_pathloss_gain_samples_max - epsilon)
                nodecounter -= 1
            else:
                break
                
    return y

def our_remove_attack_new_magnitude_based_perturbation(y, remove_perturbation_percentage):
    """
    Description for this function.

    Parameters:
        y: Type and description for y.
        remove_perturbation_percentage: Type and description for remove_perturbation_percentage.

    Returns:
        y: Type and description for the returned y.
    """
    y_temp = y.numpy()
    _, r, _ = y_temp.shape
    
    desired_pathloss_gain_samples = y_temp[0].diagonal()
    desired_pathloss_gain_samples_min = desired_pathloss_gain_samples.min()
    desired_pathloss_gain_samples_max = desired_pathloss_gain_samples.max()
    
    loc = np.where(y_temp[0] != np.multiply(y_temp[0], np.eye(r)))
    undesired_pathloss_no_gain_samples = y_temp[0][loc]
    
    remove_edge_no_samples = int(np.floor((1-remove_perturbation_percentage) * desired_pathloss_gain_samples.shape[0]))
    cntr = remove_edge_no_samples
    location = np.where(np.logical_and(y_temp[0] >= desired_pathloss_gain_samples_min, y_temp[0] <= desired_pathloss_gain_samples_max))
    remove_choices = np.random.choice(np.arange(len(location[0])), size=cntr, replace=False)
    
    newloc = np.delete(np.asarray(location), remove_choices, 1)
    newloc = tuple(map(tuple, newloc))
    y_temp[0][newloc] = y_temp[0][newloc] * 0.1

    return y

def our_add_attack_budget_constraint_new_new(y, add_perturbation_percentage):
    """
    Function to add attack with budget constraint.
    
    Parameters:
        y: Tensor
            Input tensor with certain specifications.
        add_perturbation_percentage: float
            Percentage of perturbation to be added.
    
    Returns:
        y: Tensor
            Modified input tensor.
    """
    y_temp = y.numpy()
    _, r, _ = y_temp.shape
    
    # Extracting desired and undesired pathloss gain samples and computing statistics
    desired_pathloss_gain_samples = y_temp[0].diagonal()
    desired_pathloss_gain_samples_max = desired_pathloss_gain_samples.max()
    loc = np.where(y_temp[0] != np.multiply(y_temp[0], np.eye(r)))
    undesired_pathloss_no_gain_samples = y_temp[0][loc]
    undesired_pathloss_no_gain_samples_min = undesired_pathloss_no_gain_samples.min()
    
    # Calculating the budget and initializing epsilon
    budget = add_perturbation_percentage * np.sum(desired_pathloss_gain_samples)
    epsilon = desired_pathloss_gain_samples_max
    
    # Finding minimum location and adjusting budget
    minloc = np.where(y_temp[0] == undesired_pathloss_no_gain_samples_min)
    if desired_pathloss_gain_samples_max - undesired_pathloss_no_gain_samples_min > budget:
        y_temp[0][minloc] = undesired_pathloss_no_gain_samples_min + budget
        budget = -1
    else:
        nodecounter = r
        while budget >= 0 and nodecounter >= 0:
            undesired_pathloss_no_gain_samples_min = undesired_pathloss_no_gain_samples.min()
            minloc = np.where(y_temp[0] == undesired_pathloss_no_gain_samples_min)
            if budget - (epsilon - undesired_pathloss_no_gain_samples_min) >= 0:
                y_temp[0][minloc] = epsilon
                budget = budget - (epsilon - undesired_pathloss_no_gain_samples_min)
                nodecounter -= 1
                
                # Re-fetching the undesired samples after modification
                loc = np.where(y_temp[0] != np.multiply(y_temp[0], np.eye(r)))
                undesired_pathloss_no_gain_samples = y_temp[0][loc]
            else:
                break
    
    return y

def our_add_attack_new_new(y, add_perturbation_percentage):
    """
    Function to add attack with a new method.
    
    Parameters:
        y: Tensor
            Input tensor with certain specifications.
        add_perturbation_percentage: float
            Percentage of perturbation to be added.
    
    Returns:
        y: Tensor
            Modified input tensor.
    """
    y_temp = y.numpy()
    _, r, _ = y_temp.shape
    
    # Extracting desired and undesired pathloss gain samples and computing statistics
    desired_pathloss_gain_samples = y_temp[0].diagonal()
    desired_pathloss_gain_samples_max = desired_pathloss_gain_samples.max()
    loc = np.where(y_temp[0] != np.multiply(y_temp[0], np.eye(r)))
    undesired_pathloss_no_gain_samples = y_temp[0][loc]
    undesired_pathloss_no_gain_samples_min = undesired_pathloss_no_gain_samples.min()
    undesired_pathloss_no_gain_samples_max = undesired_pathloss_no_gain_samples.max()
    
    # Determining the number of samples to add
    add_edge_no_samples = int(np.floor((1-add_perturbation_percentage) * undesired_pathloss_no_gain_samples.shape[0]))
    cntr = add_edge_no_samples
    
    # Finding the locations to be affected
    location = np.where(np.logical_and(y_temp[0] >= undesired_pathloss_no_gain_samples_min, y_temp[0] <= undesired_pathloss_no_gain_samples_max))
    add_choices = np.random.choice(np.arange(len(location[0])), size=cntr, replace=False)
    
    # Applying the perturbation
    newloc = np.delete(np.asarray(location), add_choices, 1)
    a = int(np.ceil(add_perturbation_percentage * 49 * 50))
    newloc = newloc[:, :a]
    y_temp[0][newloc] = desired_pathloss_gain_samples_max
    
    return y


class InitParameters():
    """Class to initialize wireless network parameters.
    
    Attributes are related to physical and environmental properties 
    of the wireless network such as bandwidth, carrier frequency, 
    transmission power, noise, etc.
    """
    def __init__(self):
        # wireless network settings
        self.n_links = train_K  # Number of links in the training set
        self.field_length = 1000  # Physical length of the network field
        # Direct link lengths (limits)
        self.shortest_directLink_length = 10
        self.longest_directLink_length = 50
        self.shortest_crossLink_length = 1  # Shortest cross link length
        self.bandwidth = 5e6  # Bandwidth in Hz
        self.carrier_f = 2.4e9  # Carrier frequency in Hz
        # Transmitter and Receiver heights in meters
        self.tx_height = 1.5
        self.rx_height = 1.5
        # Antenna gain and transmission power
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        # Convert transmission power from dB to Watts
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        # Noise power and SNR gap settings
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 10
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)
        # Descriptive string for settings
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5  # Cell length in the occupancy grid
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)  # Number of grid cells

def normalize_data(train_data, test_data):
    """Function to normalize training and testing data.
    
    Parameters:
        train_data: ndarray
            Training data matrix.
        test_data: ndarray
            Testing data matrix.
            
    Returns:
        norm_train: ndarray
            Normalized training data.
        norm_test: ndarray
            Normalized testing data.
    """
    mask = np.eye(train_K)
    train_copy = np.copy(train_data)
    # Separate and normalize diagonal and off-diagonal elements in training data
    diag_H = np.multiply(mask, train_copy)
    diag_mean = np.sum(diag_H) / (train_layouts * train_K)
    diag_var = np.sqrt(np.sum(np.square(diag_H)) / (train_layouts * train_K))
    tmp_diag = (diag_H - diag_mean) / diag_var
    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag) / (train_layouts * train_K / (train_K - 1))
    off_diag_var = np.sqrt(np.sum(np.square(off_diag)) / (train_layouts * train_K / (train_K - 1)))
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)
    norm_train = np.multiply(tmp_diag, mask) + tmp_off_diag

    # Repeat the normalization for testing data
    mask = np.eye(test_K)
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask, test_copy)
    tmp_diag = (diag_H - diag_mean) / diag_var
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)
    norm_test = np.multiply(tmp_diag, mask) + tmp_off_diag
    
    return norm_train, norm_test

def reshape_modified_adjacency(modified_adj):
    """Function to reshape the modified adjacency matrix into edge index format.
    
    Parameters:
        modified_adj: ndarray
            Binary adjacency matrix with shape [num_nodes, num_nodes]
            
    Returns:
        edge_index: ndarray
            2D array of shape [2, num_edges] containing source and target node indices for edges.
    """
    # Identify where in the matrix the value is 1
    one_locations = np.argwhere(modified_adj == 1)
    edge_index = np.zeros((2, len(one_locations)))

    for i in range(len(one_locations)):
        edge_index[0, i], edge_index[1, i] = one_locations[i][0], one_locations[i][1]
    
    return edge_index

class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def reset_parameters(self):
        # Ensure your reset function is defined or use appropriate PyTorch functions
        # reset(self.mlp1)
        # reset(self.mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :1], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        # Your handling of dimensions should be validated
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg
    
    def __repr__(self):
        return '{}(mlp1={}, mlp2={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)

def MLP(channels, batch_norm=True):
    # Ensure batch_norm is utilized as needed
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i])
        for i in range(1, len(channels))
    ])

class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()
        self.mlp1 = MLP([3, 32, 32])
        self.mlp2 = MLP([34, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1), Sigmoid())])
        self.conv = IGConv(self.mlp1, self.mlp2)
    
    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        #x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        #x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out

train_K = 20
test_K = 20
train_layouts = 2000
test_layouts = 500

train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)
train_path_losses = wg.compute_path_losses(train_config, train_dists)

test_config = init_parameters()
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)

norm_train, norm_test = normalize_data(1/train_dists, 1/test_dists)

directLink_channel_losses = helper_functions.get_directLink_channel_losses(test_path_losses)
crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(test_path_losses)

Y1 = FP_optimize(test_config, test_path_losses,np.ones([test_layouts, test_K]))
rates1 = helper_functions.compute_rates(test_config,
            Y1, directLink_channel_losses, crossLink_channel_losses)
sr1 = np.mean(np.sum(rates1,axis=1))

Pini = np.random.rand(test_layouts, test_K, 1)
Y2 = wf.batch_WMMSE2(Pini,np.ones([test_layouts, test_K]),np.sqrt(test_path_losses), 1, var)
rates2 = helper_functions.compute_rates(test_config,
            Y2, directLink_channel_losses, crossLink_channel_losses)
sr2 = np.mean(np.sum(rates2,axis=1))

print('FPLinQ:',sr1)
print('WMMSE:', sr2)

train_data_list = proc_data(train_path_losses, train_dists, norm_train, train_K, 0)
test_data_list = proc_data(test_path_losses, test_dists, norm_test, test_K, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

def sr_loss(data, out, K):
    power = out[:,1]
    power = torch.reshape(power, (-1, K, 1))

    abs_H_2 = data.y
    abs_H_2 = abs_H_2.permute(0,2,1)
    rx_power = torch.mul(abs_H_2, power)

    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sum_rate)
    return loss


def sr_loss_revised(data, out, K, remove_perturbation_percentage=0.7):
    power = out[:, 1].view(-1, K, 1)
    abs_H_2 = data.y.permute(0, 2, 1)
    rx_power = abs_H_2 * power
    
    mask = torch.eye(K, device=out.device)
    valid_rx_power = (rx_power * mask).sum(dim=1)
    interference = (rx_power * (1 - mask)).sum(dim=1) + var
    rate = torch.log2(1 + valid_rx_power / interference)
    
    # Assuming you need the following computations for logging or monitoring
    A = rate.detach().clone()
    r, c = A.shape
    ind = torch.argsort(A, dim=1)
    m = int(c * remove_perturbation_percentage)
    
    q = A.gather(1, ind[:, -m:]).mean(dim=1)
    p = A.gather(1, ind[:, :-m]).mean(dim=1)
    k = (A.sum(dim=1) - p * (c - m)) / A.sum(dim=1)
    
    rate_mean_max = q.mean()
    rate_mean_rest = p.mean()
    rate_maximum_usage = k.mean()
    
    # Implement any logging/usage of rate_mean_max, rate_mean_rest, rate_maximum_usage here
    
    sum_rate = rate.sum(dim=1).mean()
    loss = -sum_rate
    return loss

def train(model, optimizer, train_loader, train_K, train_layouts, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data, out, train_K)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts

def test(model, test_loader, loss_fn, test_K, test_layouts, device, log_time=False):
    model.eval()
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            start = time.time()
            out = model(data)
            end = time.time()
            if log_time:
                print('CGCNet time', end-start)
            loss = loss_fn(data, out, test_K)
            print(f'loss is {loss}')
            total_loss += loss.item() * data.num_graphs
    return total_loss / test_layouts

def test_revised(model, test_loader, loss_function, test_K, test_layouts, device):
    model.eval()
    total_loss = 0
    
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            start_time = time.time()
            output = model(data)
            elapsed_time = time.time() - start_time
            print(f'CGCNet inference time: {elapsed_time:.4f} seconds')
            
            loss = loss_function(data, output, test_K)
            print(f'Loss: {loss.item():.4f}')
            
            total_loss += loss.item() * data.num_graphs
            
    average_loss = total_loss / test_layouts
    return average_loss

def simple_greedy(X, AAA, label, threshold_k):
    n = X.shape[0]
    thd = int(np.sum(label) / n)
    Y = np.zeros((n, threshold_k))
    for i in range(n):
        alpha = AAA[i, :]
        H_diag = alpha * np.square(np.diag(X[i, :, :]))
        indices_to_update = np.argsort(H_diag)[::-1][:thd]
        Y[i, indices_to_update] = 1
    return Y

train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)

for epoch in tqdm(range(1, 200)):
    loss_train = train(model, optimizer, train_loader, train_K, train_layouts, device)
    if epoch % 4 == 0:
        loss_test = test(model, test_loader, sr_loss, test_K, test_layouts, device, log_time=False)
        print(f'Epoch {epoch:03d}, Train Loss: {loss_train:.4f}, Test Loss: {loss_test:.4f}')
    scheduler.step()

def test_scenario(gen_tests, init_parameters_func, generate_layouts_func, compute_path_losses_func, compute_rates_func, 
                  get_directLink_channel_losses_func, get_crossLink_channel_losses_func, FP_optimize_func, 
                  batch_WMMSE2_func, simple_greedy_func, normalize_data_func, proc_data_func, test_revised_func, 
                  train_dists, var, device):

    density = test_config.field_length**2 / test_K
    
    for test_K in gen_tests:
        test_layouts = 20
        test_config = init_parameters_func()
        test_config.n_links = test_K
        
        field_length = int(np.sqrt(density * test_K))
        test_config.field_length = field_length
        
        layouts, test_dists = generate_layouts_func(test_config, test_layouts)
        test_path_losses = compute_path_losses_func(test_config, test_dists)
        
        directLink_channel_losses = get_directLink_channel_losses_func(test_path_losses)
        crossLink_channel_losses = get_crossLink_channel_losses_func(test_path_losses)
        
        start = time.time()
        Y = FP_optimize_func(test_config, test_path_losses, np.ones([test_layouts, test_K]))
        print('FP time:', time.time() - start)
        
        Pini = np.random.rand(test_layouts, test_K, 1)
        start = time.time()
        Y2 = batch_WMMSE2_func(Pini, np.ones([test_layouts, test_K]), np.sqrt(test_path_losses), 1, var)
        print('WMMSE time:', time.time() - start)
        
        rates1 = compute_rates_func(test_config, Y, directLink_channel_losses, crossLink_channel_losses)
        rates2 = compute_rates_func(test_config, Y2, directLink_channel_losses, crossLink_channel_losses)
        sr1 = np.mean(np.sum(rates1, axis=1))
        sr2 = np.mean(np.sum(rates2, axis=1))
        print('FPlinQ:', sr1)
        print('WMMSE:', sr2)
        
        bl_Y = simple_greedy_func(test_path_losses, np.ones([test_layouts, test_K]), Y)
        rates_bl = compute_rates_func(test_config, bl_Y, directLink_channel_losses, crossLink_channel_losses)
        sr_bl = np.mean(np.sum(rates_bl, axis=1))
        print('baseline:', sr_bl)
        
        _, norm_test = normalize_data_func(1/train_dists, 1/test_dists)
        test_data_list = proc_data_func(test_path_losses, test_dists, norm_test, test_K, 1)
        test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
        
        loss2 = test_revised_func()
        print('CGCNet:', loss2)
        print('performance:', -loss2/sr2)

# 'main' function to encapsulate script functionality
if __name__ == "__main__":
    install_pytorch_geometric_dependencies()
    mount_google_drive()

    # Further explanations to be fully understood by others.
    wg = reload(wg)  # Reloading: Explain why this is necessary.
    FPLinQ = reload(FPLinQ)  # Reloading: Elaborate on purpose and impact.
    helper_functions = reload(helper_functions)  # Reloading: Detail reasons and implications.

    # Your main functionalities and script operations go here.
    total_loss / train_layouts = train(model, test_loader, loss_fn, test_K, test_layouts, device)
    total_loss / test_layouts = test(model, test_loader, loss_fn, test_K, test_layouts, device)

    test_scenario(gen_tests, init_parameters_func, generate_layouts_func, compute_path_losses_func, compute_rates_func, 
                  get_directLink_channel_losses_func, get_crossLink_channel_losses_func, FP_optimize_func, 
                  batch_WMMSE2_func, simple_greedy_func, normalize_data_func, proc_data_func, test_revised_func, 
                  train_dists, var, device)
