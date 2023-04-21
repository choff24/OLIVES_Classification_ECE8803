import numpy as np
import torch.nn as nn
import torch.utils.data
import dataloader
from torchvision import transforms
import os
import Models
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained = False
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
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])

args = dataloader.parse_args()
args.data_root = os.getcwd()
trainset = dataloader.OCTDataset(args, 'train', transform=transform)
testset = dataloader.OCTDataset(args, 'test', transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False)
model.conv1
if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/model.pth'))

model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=2.5e-3,
                             weight_decay=1e-6)

epochs = 100
outputs = []
total_loss = []

for epoch in range(epochs):
    tot_loss = 0
    for iteration, (image, label) in enumerate(train_loader):

        #image = image.reshape(-1, 224*224).to(device)
        image = image.to(device)
        label = label.to(device)
        reconstructed = model(image)

        loss = loss_function(reconstructed, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.to('cpu').detach().numpy()

    print('Epoch ' + str(epoch) + ' Completed Loss: ' + str(tot_loss/iteration))
    total_loss.append(tot_loss)

    if epoch % 100 == 0:
        torch.save(model.state_dict(), os.getcwd() + '/GoogleNet_Model.pth')


torch.save(model.state_dict(), os.getcwd() + '/model.pth')

for (image, _) in test_loader:
    image = image.reshape(-1, 224*224).to(device)
    reconstructed = model(image)

    plt.figure()
    plt.imshow(image.to('cpu').reshape(224, 224).numpy())
    plt.figure()
    plt.imshow(reconstructed.to('cpu').reshape(224, 224).detach().numpy())
    plt.show()


