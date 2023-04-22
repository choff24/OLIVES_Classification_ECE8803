import numpy as np
import torch.utils.data
import dataloader_3D as dataloader
from torchvision import transforms
import os
import Models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchio as tio

# Run the convolutional autoencoder used in the final project

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
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

# Setup image transformations, normalization, and augmentation
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

spatial = tio.OneOf({
    tio.RandomAffine(): 1,
    tio.RandomElasticDeformation(): 0,
    },
    p=0.75
)

ThreeDimTransform = tio.Compose([spatial])

args = dataloader.parse_args()
args.data_root = os.getcwd()
args.ThreeDim = ThreeDimTransform
trainset = dataloader.OCTDataset(args, 'train', transform=transform)
testset = dataloader.OCTDataset(args, 'test', transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=2, shuffle=True)

# Load pretrained weights if wanted
model = Models.CAE_3D()
pretrained = False

if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/Models/3D_CAE_Model.pth'))

model.to(device)
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),
                            lr=1e-3,
                            weight_decay=1e-6,)

epochs = 50
outputs = []
total_loss = []
train = True
plot = False
num_plot = 10
weights = torch.Tensor([1/159, 1/240, 1/96])

# Training loop
if train:

    # Setup oversampler using a weighted random sampler to help with the imbalanced data
    trainset = dataloader.OCTDataset(args, 'train', transform=transform)

    targets = trainset._labels

    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

    train_sub = torch.utils.data.Subset(trainset, train_idx)
    val_sub = torch.utils.data.Subset(trainset, val_idx)

    reciprocal_weights = np.zeros(len(train_sub))

    for i, index in enumerate(trainset._labels[train_sub.indices]):
        reciprocal_weights[i] = weights[index]

    sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(reciprocal_weights), len(train_sub), replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_sub, batch_size=3, shuffle=False, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=3, shuffle=False)

    train_loss_hist = []
    val_loss_hist = []

    # Actual training loop
    for epoch in range(epochs):

        train_loss = 0
        test_loss = 0
        val_loss = 0

        for iteration, (image, label) in enumerate(train_loader):

            with torch.no_grad():
                image = normalize(image)

            optimizer.zero_grad()
            image = image.to(device)
            reconstructed = model(image)

            loss = loss_function(reconstructed, image)
            loss.backward()
            optimizer.step()

            train_loss += loss.to('cpu').detach().numpy()

        for iteration_val, (image_val, label_val) in enumerate(val_loader):

            with torch.no_grad():
                image_val = normalize(image_val)
                image_val = image_val.to(device)
                reconstructed_val = model(image_val)
                loss_val = loss_function(reconstructed_val, image_val)
                val_loss += loss_val.to('cpu').detach().numpy()

        print('Epoch ' + str(epoch) + ' --- Train Loss: ' + str(train_loss) + ' --- Train MSE: '
              + str(train_loss/(iteration+1)) + ' --- Validation Loss: ' + str(val_loss) + ' --- Train MSE: '
              + str(val_loss/(iteration_val+1)))

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.getcwd() + '/Models/3D_CAE_Model.pth')

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)


    train_loss_hist = np.array(train_loss_hist)
    val_loss_hist = np.array(val_loss_hist)

    train_loss_hist /= train_loss_hist[0]
    val_loss_hist /= val_loss_hist[0]
    plt.figure()
    plt.plot(range(epochs), train_loss_hist, 'r')
    plt.plot(range(epochs), val_loss_hist, 'b')
    plt.legend(('Train Loss', 'Validation Loss'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convolutional Autoencoder Loss')
    plt.savefig(os.getcwd() + '/Plots/CAE_3D_Loss.png')

# Makes a picture of the reconstructed image to compare to the original image. Was originally made for 2D data so have
# not reused on 3D model since
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







