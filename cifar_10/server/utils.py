# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch CIFAR-10 image classification.
The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223


from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import densenet121, mobilenet_v2
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
DATA_ROOT = Path("/lung80")
#DATA_ROOT = Path("/Users/ahmad/fed_rpp/my_data")


# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def ResNet18():
    """Returns a ResNet18 model from TorchVision adapted for CIFAR-10."""

    model = resnet18(num_classes=3)

    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    model.maxpool = torch.nn.Identity()

    return model

def ResNet50():
    """Returns a ResNet50 model from TorchVision adapted for CIFAR-10."""

    model = resnet50(num_classes=3)
    
    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    model.maxpool = torch.nn.Identity()
    
    return model

def DenseNet121():
    """Returns a DenseNet121 model from TorchVision adapted for CIFAR-10."""
    model = densenet121(num_classes=3)

    # replace w/ smaller input layer
    model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features.conv0.weight, mode="fan_out", nonlinearity="relu")

    return model


def MobileNetV2():
    """Returns a MobileNetV2 model from TorchVision adapted for CIFAR-10."""
    model = mobilenet_v2(num_classes=3)

    # replace w/ smaller input layer
    model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features[0][0].weight, mode="fan_out", nonlinearity="relu")

    return model

def EfficientNetB0():
    """Returns an EfficientNetB0 model from efficientnet_pytorch adapted for CIFAR-10."""
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

    # replace w/ smaller input layer
    model._conv_stem = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model._conv_stem.weight, mode="fan_out", nonlinearity="relu")

    return model

def load_model(model_name: str) -> nn.Module:
    if model_name == "Net":
        return Net()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "DenseNet121":
        return DenseNet121()
    elif model_name == "MobileNetV2":
        return MobileNetV2()
    elif model_name == "EfficientNetB0":
        return EfficientNetB0()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")



# To read files from my machine directory.
def load_cifar():
    transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    trainset = datasets.ImageFolder(DATA_ROOT/"train", transform=transform)
    testset = datasets.ImageFolder(DATA_ROOT/"test", transform=transform)
    return trainset, testset
"""

def load_cifar(download=True) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    training_set = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testing_set = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    classes = torch.tensor([0, 1, 2, 3, 4])

    indices_t = (torch.tensor(training_set.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    indices_s = (torch.tensor(testing_set.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]

    trainset = torch.utils.data.Subset(training_set, indices_t)
    testset = torch.utils.data.Subset(testing_set, indices_s)

    return trainset, testset
"""
def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print(f"Epoch took: {time() - t:.2f} seconds")


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
