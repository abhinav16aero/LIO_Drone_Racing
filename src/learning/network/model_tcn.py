"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/model_tcn.py
"""


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


dict_activation = {"ReLU": nn.ReLU, "GELU": nn.GELU}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


# class TemporalBlock(nn.Module):
#     def __init__(
#         self,
#         n_inputs,
#         n_outputs,
#         kernel_size,
#         stride,
#         dilation,
#         padding,
#         dropout=0.2,
#         activation=nn.ReLU,
#     ):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(
#             nn.Conv1d(
#                 n_inputs,
#                 n_outputs,
#                 kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#             )
#         )
#         self.chomp1 = Chomp1d(padding)
#         self.activation1 = activation()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = weight_norm(
#             nn.Conv1d(
#                 n_outputs,
#                 n_outputs,
#                 kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#             )
#         )
#         self.chomp2 = Chomp1d(padding)
#         self.activation2 = activation()
#         self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(
#             self.conv1,
#             self.chomp1,
#             self.activation1,
#             self.dropout1,
#             self.conv2,
#             self.chomp2,
#             self.activation2,
#             self.dropout2,
#         )
#         self.downsample = (
#             nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         )
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)


# class TemporalConvNet(nn.Module):
#     def __init__(
#         self,
#         num_inputs,
#         num_hidden_channels,
#         kernel_size=2,
#         dropout=0.2,
#         activation="ReLU",
#     ):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_hidden_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1]
#             out_channels = num_hidden_channels[i]
#             layers += [
#                 TemporalBlock(
#                     in_channels,
#                     out_channels,
#                     kernel_size,
#                     stride=1,
#                     dilation=dilation_size,
#                     padding=(kernel_size - 1) * dilation_size,
#                     dropout=dropout,
#                     activation=activation,
#                 )
#             ]

#         # print("receptive field = ", 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         # only return last component
#         return self.network(x)


# class Tcn(nn.Module):
#     """
#     This tcn is trained so that the output at current time is a vector that contains
#     the prediction using the last second of inputs.
#     The receptive field is givent by the input parameters.
#     """

#     def __init__(
#         self,
#         input_size,
#         output_size,
#         num_channels,
#         kernel_size,
#         dropout,
#         activation="ReLU",
#     ):
#         super(Tcn, self).__init__()
#         self.tcn = TemporalConvNet(
#             input_size,
#             num_channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#             activation=dict_activation[activation],
#         )
#         self.linear = nn.Linear(num_channels[-1], output_size)
#         self.init_weights()

#     def init_weights(self):
#         self.linear.weight.data.normal_(0, 0.01)

#     def get_num_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

#     def forward(self, x):
#         x = self.tcn(x)
#         x = self.linear(x[:, :, -1])
#         return x



# SE Block for channel recalibration
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()
        se = self.global_avg_pool(x).view(batch_size, channels)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se)).view(batch_size, channels, 1)
        return x * se.expand_as(x)

# Temporal Block with Multi-Scale Feature Aggregation and SE Block
class MultiScaleTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_sizes,  # list of integers, e.g., [2, 4, 8]
        stride,
        padding,
        dilation,
        dropout=0.2,
        activation=nn.ReLU,
    ):
        super(MultiScaleTemporalBlock, self).__init__()
        
        # Ensure kernel_sizes is iterable (list or tuple)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        
        # Calculate padding dynamically to ensure consistent output dimensions
        self.convs = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size=k,
                    stride=stride,
                    padding=((k - 1) * dilation) // 2,  # `same` padding logic
                    dilation=dilation,
                )
            )
            for k in kernel_sizes
        ])
        
        self.se_blocks = nn.ModuleList([
            SqueezeExcitationBlock(n_outputs) for _ in kernel_sizes
        ])

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total output channels based on the number of kernels
        total_output_channels = n_outputs * len(kernel_sizes)
        self.downsample = nn.Conv1d(n_inputs, total_output_channels, 1) if n_inputs != total_output_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for conv in self.convs:
            conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Apply each convolution and squeeze-and-excitation block
        conv_outputs = [se(conv(x)) for conv, se in zip(self.convs, self.se_blocks)]
        
        # Concatenate along the channel dimension
        out = torch.cat(conv_outputs, dim=1)
        
        # Apply downsampling if necessary
        res = x if self.downsample is None else self.downsample(x)
        
        if out.size(2) != res.size(2):
            min_dim = min(out.size(2), res.size(2))
            out = out[:, :, :min_dim]
            res = res[:, :, :min_dim]
        
        return self.relu(out + res)


# Updated TCN model with Multi-Scale Temporal Block
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden_channels,
        kernel_size=2,
        dropout=0.2,
        activation="ReLU",
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1] * len(kernel_size)
            out_channels = num_hidden_channels[i]
            layers += [
                MultiScaleTemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(max(kernel_size) - 1) * dilation_size,
                    dropout=dropout,
                    activation=dict_activation[activation],
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# TCN model wrapper with output linear layer
class Tcn(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        activation="ReLU",
    ):
        super(Tcn, self).__init__()
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size,
            dropout=dropout,
            activation=activation,
        )
        self.linear = nn.Linear(num_channels[-1] * len(kernel_size), output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x