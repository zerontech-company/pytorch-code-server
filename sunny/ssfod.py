from cv2 import transform
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
from PIL import Image

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

# Construct argument parser
#ap = argparse.ArgumentParser()
#ap.add_argument("--mode", required=True, help="Training mode: finetune/transfer/scratch")
#args= vars(ap.parse_args())

# Set training mode
train_mode='finetune' #args["mode"]

# Set the train and validation directory paths
train_directory =  '/projects/data/train'
valid_directory = '/projects/data/train'
# Set the model save path
PATH="zerontech_resnext50_32x4d_small_size_finetune.pth" 

# Batch size
bs = 16
# Number of epochs
num_epochs = 50
# Number of classes
# num_classes = 16
# Number of workers
num_cpu = multiprocessing.cpu_count()
num_cpu = 1
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
        transforms.Resize(size=(256,256)),
        transforms.ToTensor()
    ]),

    'valid': transforms.Compose([
        transforms.Resize(size=(256, 256)),
      #  transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
}
 
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# Class names or target labels
class_names = dataset['train'].classes
num_classes=len(class_names)
print(len(class_names))
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("gpu device : ", device)
if train_mode=='finetune':
    # Load a pretrained model - Resnet18
    print("\nLoading resnext50_32x4d for finetuning ...\n")
    # model_ft = models.resnet18(pretrained=True)
    model_ft = models.resnext50_32x4d(pretrained=True)
    

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes )

elif train_mode=='scratch':
    # Load a custom model - VGG11
    print("\nLoading VGG11 for training from scratch ...\n")
    model_ft = MyVGG11(in_ch=3, num_classes=11)

    # Set number of epochs to a higher value
    num_epochs=100

elif train_mode=='transfer':
    # Load a pretrained model - MobilenetV2
    print("\nLoading resnext50_32x4d as feature extractor ...\n")
    # model_ft = models.resnet18(pretrained=True)    
    model_ft =models.resnext50_32x4d(pretrained=True)

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-11]:
        params.requires_grad = False
   
        # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.out_features
    # num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        )    


    # import torchsummary
    # # Modify fc layers to match num_classes
    # print(torchsummary.summary(model_ft, (3, 256, 256),device='cpu'))
    # num_ftrs=model_ft.classifier[-1].in_features
    # model_ft.classifier=nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=False),
    #     nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    #     )    

# Transfer the model to GPU
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )
summary(model_ft, input_size=(3, 512, 512))
print(model_ft)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, PATH.replace('.pth','{}.pth'.format(epoch)))
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, PATH.replace('.pth','{}.pth'.format(epoch)))
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)

'''
Sample run: python train.py --mode=finetue
'''
