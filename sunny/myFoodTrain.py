
from typing import Dict
from dagster import get_dagster_logger, job, op, In, DagsterType

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

"""Import Aim"""

from aim import Run

"""Load the dataset"""

PATH="zerontech_resnext50_32x4d_small_size_finetune.pth" 

class myFoodClassification:
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

        train_mode='finetune'

        # Set the train and validation directory paths
        train_directory =  '/projects/data/train'
        valid_directory =  '/projects/data/train'
        # Set the model save path
        #PATH="zerontech_resnext50_32x4d_small_size_finetune.pth" 

        # Batch size
        bs = 32
        # Number of epochs
        num_epochs = 50
        #num_cpu = multiprocessing.cpu_count()
        num_cpu = 20
        # Applying transforms to the data
        image_transforms = { 
            'train': transforms.Compose([
            #  transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

                # transforms.RandomHorizontalFlip(),
            #  transforms.CenterCrop(size=224),
                # transforms.Normalize([0.485, 0.456, 0.406],
                #                      [0.229, 0.224, 0.225]),
                transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
                transforms.Resize(size=(512,512)),
                transforms.ToTensor()
            ]),

            'valid': transforms.Compose([
                transforms.Resize(size=(512, 512)),
            #  transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406],
                #                      [0.229, 0.224, 0.225])
            ])
        }
        

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

@op
def doTrain(hyper):

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
    aim_run = Run(experiment='FoodObjectDetection')

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
