import numpy as np
import torch.utils.data
import dataloader
from torchvision import transforms
import os
import Models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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



transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])

args = dataloader.parse_args()
args.data_root = os.getcwd()
trainset = dataloader.OCTDataset(args, 'train', transform=transform_train)
testset = dataloader.OCTDataset(args, 'test', transform=transform_test)


test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=True)

model = Models.CAE2()
pretrained = False

if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/CAE_Model.pth'))

model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=2.5e-3,
                             weight_decay=1e-6)

epochs = 100
outputs = []
total_loss = []
train = True
plot = False
num_plot = 10
weights = torch.Tensor([1/7788, 1/11760, 1/4704])

if train:

    trainset = dataloader.OCTDataset(args, 'train', transform=transform_train)

    targets = np.zeros(len(trainset))

    for index, x in enumerate(trainset):
        targets[index] = x[1]

    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

    train_sub = torch.utils.data.Subset(trainset, train_idx)
    val_sub = torch.utils.data.Subset(trainset, val_idx)

    reciprocal_weights = np.zeros(len(train_sub))
    for index, x in enumerate(train_sub):
        reciprocal_weights[index] = weights[x[1]]

    val_sub.dataset.transform = transform_test
    sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(reciprocal_weights), len(train_sub), replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_sub, batch_size=128, shuffle=False, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=128, shuffle=False)

    for epoch in range(epochs):

        train_loss = 0
        test_loss = 0
        val_loss = 0

        for iteration, (image, label) in enumerate(train_loader):

            optimizer.zero_grad()

            image = image.to(device)
            reconstructed = model(image)

            loss = loss_function(reconstructed, image)
            loss.backward()
            optimizer.step()

            train_loss += loss.to('cpu').detach().numpy()

        for iteration_val, (image_val, label_val) in enumerate(val_loader):

            with torch.no_grad():

                image_val = image_val.to(device)
                reconstructed_val = model(image_val)
                loss_val = loss_function(reconstructed_val, image_val)
                val_loss += loss_val.to('cpu').detach().numpy()

        print('Epoch ' + str(epoch) + ' --- Train Loss: ' + str(train_loss) + ' --- Train MSE: '
              + str(train_loss/(iteration+1)) + ' --- Validation Loss: ' + str(val_loss) + ' --- Train MSE: '
              + str(val_loss/(iteration_val+1)))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.getcwd() + '/CAE2_Model_Large.pth')

if plot:
    for iteration, (image, label) in enumerate(test_loader):
        if iteration > num_plot:
            break
        else:
            i = 0
            image = image.to(device)
            reconstructed = model(image)
            #reconstructed = normalize(reconstructed)
            im = image[i, 0, :, :].to('cpu').detach().numpy()
            im_r = reconstructed[i, 0, :, :].to('cpu').detach().numpy()
            #low_level_r = np.argmax(model.get_representation(image)[i, :, :, :].to('cpu').detach().numpy(), axis=0)
            plt.figure()
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(im)
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(im_r)
            #ax3 = plt.subplot(1, 3, 3)
            #ax3.imshow(low_level_r)
            #ax3.set_title(label[i])

    plt.show()







