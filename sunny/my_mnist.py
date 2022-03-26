
from typing import Dict
from dagster import get_dagster_logger, job, op, In, DagsterType

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

"""Import Aim"""

from aim import Run

# Hyper parameters
"""
batch_size = 50
num_classes = 10
learning_rate = 0.01
num_epochs = 20
"""

"""Load the dataset"""

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def is_list_of_dicts(_, value):
    return isinstance(value, list)

SimpleDataFrame = DagsterType(
    name="SimpleDataFrame",
    type_check_fn=is_list_of_dicts,
    description="A naive representation of a data frame, e.g., as returned by csv.DictReader.",
)

#@op(ins={"hyper": In(SimpleDataFrame)})
#@op(ins={"hyper": In(SimpleDataFrame)})
@op
def doTrainMNIST(hyper):

    #global batch_size, num_classes, learning_rate, num_epochs
    # Hyper parameters

    #print( hyper )

    batch_size = hyper["batch_size"]
    num_classes = hyper["num_classes"]
    learning_rate = hyper["learning_rate"]
    num_epochs = hyper["num_epochs"]

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    """Build CNN model"""


    """Initialize model and optimizer"""

    model = ConvNet(num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """Train the model and track metrics and params with Aim"""

    # Initialize a new Aim run
    aim_run = Run(experiment='sunny_mnist5')

    # aim - Track hyper parameters
    aim_run['hparams'] = {
        'num_epochs': num_epochs,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], '
                    'Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                            total_step, loss.item()))

                # aim - Track model loss function
                aim_run.track(loss.item(), name='loss', epoch=epoch,
                            context={'subset':'train'})

                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total

                # aim - Track metrics
                aim_run.track(acc, name='accuracy', epoch=epoch, context={'subset': 'train'})
                
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 5 == 0:
                acc = 100 * correct / total
                aim_run.track(acc, name='accuracy', context={'subset': 'test'})
        
    aim_run.finalize()

    """## Read, export, visualize



    """
