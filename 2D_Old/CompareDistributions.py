import numpy as np
import torch.nn as nn
import torch.utils.data
import dataloader
from torchvision import transforms
import os
import Models
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}

mean = (.1706)
std = (.2112)

normalize = transforms.Normalize(mean=mean, std=std)

# normally 224x224
#transform = transforms.Compose([
#    transforms.Resize(size=(224, 224)),
#    transforms.ToTensor(),
    #normalize
#])

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    #transforms.CenterCrop((170, 224)),
    #transforms.Pad((0, 27, 0, 27), 0),
    transforms.ToTensor()
    #normalize
])

#transform_recon = transforms.Compose([normalize])
args = dataloader.parse_args()
args.data_root = os.getcwd()
trainset = dataloader.OCTDataset(args, 'train', transform=transform)
testset = dataloader.OCTDataset(args, 'test', transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=50, shuffle=True)

model = Models.CAE()
pretrained = False

if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/CAE_Model.pth'))

model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=2.5e-3,
                             weight_decay=1e-6)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 100
outputs = []
total_loss = []
train = True
plot = False
num_plot = 10

if train:
    for epoch in range(epochs):

        train_loss = 0
        test_loss = 0

        for iteration, (image_train, label_train) in enumerate(train_loader):



            for iteration, (image_test, label_test) in enumerate(test_loader):

                test0 = image_test[label_test == 0, :, :, :].flatten()
                test1 = image_test[label_test == 1, :, :, :].flatten()
                test2 = image_test[label_test == 2, :, :, :].flatten()
                train0 = image_train[label_train == 0, :, :, :].flatten()
                train1 = image_train[label_train == 1, :, :, :].flatten()
                train2 = image_train[label_train == 2, :, :, :].flatten()

                plt.figure()
                plt.hist(test0)
                plt.hist(train0)

                plt.figure()
                plt.hist(test1)
                plt.hist(train1)

                plt.figure()
                plt.hist(test2)
                plt.hist(train2)

                plt.show()




