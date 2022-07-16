# imports
import os
import tqdm
import torch
import torch.nn as nn
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor


# general config
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

random_seed = 42
torch.manual_seed(random_seed); # set a random seed for replicable results


# defining our model
class CNNModel(nn.Module):
    def __init__(self, output_size):
        super(CNNModel, self).__init__()
        self.input_layer = nn.Conv2d(3, 25, kernel_size=3, padding=1)
        self.maxpool_layer = nn.MaxPool2d(2, 2)
        self.conv_layer_2 = nn.Conv2d(25, 50, kernel_size=3, padding=1)
        self.conv_layer_3 = nn.Conv2d(50, 125, kernel_size=3, padding=1)
        self.conv_layer_4 = nn.Conv2d(125, 100, kernel_size=3, padding=1)
        self.conv_layer_5 = nn.Conv2d(100, 150, kernel_size=3, padding=1)
        self.conv_layer_6 = nn.Conv2d(150, 150, kernel_size=3, padding=1)
        self.conv_layer_7 = nn.Conv2d(150, 25, kernel_size=3, padding=1)
        self.linear_layer_1 = nn.Linear(25*127*75, 1024)
        self.linear_layer_2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_2(x)
        x = nn.functional.relu(x)

        x = self.maxpool_layer(x)

        x = self.conv_layer_3(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_4(x)
        x = nn.functional.relu()

        x = self.maxpool_layer(x)

        x = self.conv_layer_5(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_6(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_7(x)
        x = nn.functional.relu(x)

        x = x.flatten(start_dim=1)
        x = self.linear_layer_1(x)
        x = nn.functional.relu(x)
        x = self.linear_layer_2(x)
        x = nn.functional.relu(x)
        x = self.output_layer(x)

        return x


# training function from workshop 3
# will train our model over a single epoch
def train(model, train_loader, loss_fn, optimizer, device):
    model.train() # put model into training mode
    running_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader, 0): # loop through all training data
            inputs, labels = data # seperate out our inputs and outputs (labels)
            inputs, labels = inputs.to(device), labels.to(device) # put data on GPU

            # forward + backward + optimize
            optimizer.zero_grad() # clear the gradients in model parameters
            outputs = model(inputs) # forward pass, get predictions
            loss = loss_fn(outputs, labels) # calculate loss from predictions
            loss.backward() # calculate gradient wrt loss for all parameters in model that have requires_grad=True
            optimizer.step() # iterate over all parameters in the model which have requires_grad=True and update their weights

            running_loss += loss.item() # sum total loss in current epoch for printing later

            pbar.update(1) # increment progress bar
        
        return running_loss/len(train_loader) # return the total training loss for the epoch


# validation function from workshop 3
# will validate our model over a single epoch
def validation(model, val_loader, loss_fn, device):
    model.eval() # put model in validation mode
    running_loss = 0
    total = 0
    correct = 0

    with torch.no_grad(): # save memory by not saving unused gradients
        with tqdm(total=len(val_loader)) as pbar:
            for images, labels in iter(val_loader):
                images, labels = images.to(device), labels.to(device) # put data on GPU
                outputs = model(images) # pass image to model, and calculate the class probability prediction

                val_loss = loss_fn(outputs, labels) # calculates val_loss from model predictions and true labels
                running_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1) # turn the class predictions into labels
                total += labels.size(0) # sum total number of predictions
                correct += (predicted == labels).sum.item() # sum number of correct predictions

                pbar.update(1) # increment progress bar

            return running_loss/len(val_loader), correct/total # return loss and accuracy


# importing dataset and applying transforms
train_ds = ImageFolder('/data/train',
                      transform=transforms.Compose([transforms.Resize([125, 75]),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])]))

val_ds = ImageFolder('/data/test',
                      transform=transforms.Compose([transforms.Resize([125, 75]),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])]))


# creating dataloaders
batch_size = 32

training_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

num_classes = len(train_ds.classes)


# instantiate our model
model = CNNModel(num_classes)

# move the model to our GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Determine whether a GPU is available
model.to(device) # send model to GPU

# define some important parameters
loss_fn = nn.CrossEntropyLoss()
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


# perform the training loop
# training loop from workshop 3
total_epoch = 10 # Define how many epochs of training we want

# keep track of things we'd like to plot later
training_losses = []
validation_losses = []
accuracies = []

for epoch in range(total_epoch): # loops through number of epochs
    train_loss = train(model, training_loader, loss_fn, optimizer, device)  # train the model for one epoch
    val_loss, accuracy = validation(model, validation_loader, loss_fn, device) # after training for one epoch, run the validation() function to see how the model is doing on the validation dataset
  
    # keep track of interesting stuff
    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    accuracies.append(accuracy)
  
    print("Epoch: {}/{}, Training Loss: {}, Val Loss: {}, Val Accuracy: {}".format(epoch+1, total_epoch, train_loss, val_loss, accuracy))
    print('-' * 20)
    
print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'finished')


# plotting our results
# plotting results from workshop 3
plt.plot(training_losses, label="Training Losses")
plt.plot(validation_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.savefig("Losses.png")

plt.plot(accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.savefig("Accuracy.png")
