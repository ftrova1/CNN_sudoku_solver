import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./data/MNIST'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('./data/MNIST/train.csv', dtype = np.float32)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
y = df.label.values
x = df.loc[:,df.columns != "label"].values/255
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)



#Variable -> wraps Tensors and can compute backward propagation of gradients
#nn.functional contains ReLU : non-linear activation function
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data

#my image is (1,28,28)

class MyModel (nn.Module):
    def __init__(self) :
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, stride = 1, padding= 0)
        #26*26*16

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #13*13*16
        self.fcl1 = nn.Linear(2704, 64)
        self.fcl2 = nn.Linear(64, 64)
        self.fcl3 = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 13*13*16) #flatten data before feeding dense layer
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)

        return(F.log_softmax(x, dim=1))





# batch_size and epochs
batch_size = 32
EPOCHS = 3

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
test = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

model = MyModel()
# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)






# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1,1,28,28))
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels.long())

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()
    print (loss)



with torch.no_grad():
        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:

            test = Variable(images.view(-1,1,28,28))

            # Forward propagation
            outputs = model(test)
            for idx, i in enumerate (outputs):
                if torch.argmax(i).long() == labels[idx].long():
                    correct += 1

                # Total number of labels
                total += 1

print('Accuracy: {} %'.format(round(correct/total, 3)))


torch.save(model, './models/model.pth')
