import torch.utils.data
import dataloader
from torchvision import transforms
import os
import Models
import find_learning_rate
import h5py

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

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize,
    transforms.RandomRotation(180)
])

transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])

args = dataloader.parse_args()
args.data_root = os.getcwd()

trainset = dataloader.OCTDataset(args, 'train', transform=transform)
testset = dataloader.OCTDataset(args, 'test', transform=transform_test)

train_sub, val_sub = torch.utils.data.random_split(trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(1))

val_sub.dataset.transform = transform_test
train_loader = torch.utils.data.DataLoader(dataset=train_sub, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=True)

model = Models.ClassifierANN(100, 3, 10, 30)
model_dim = Models.CAE()
model_dim.load_state_dict(torch.load(os.getcwd() + '/CAE_Model.pth'))

pretrained = False

if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/ANN_Model.pth'))

model.to(device)
model_dim.to(device)

loss_function = torch.nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(),
#                            lr=1,
#                            momentum=0.9,
#                            nesterov=True)

optimizer = torch.optim.NAdam(model.parameters(),
                              lr = 2e-3,
                              weight_decay=1e-6)
min_lr = 1e-7
max_lr = 1.5
epochs = 5
num_points = 20
lr_optim = find_learning_rate.find_learning_rate(min_lr, max_lr, num_points, epochs, model, optimizer, loss_function,
                                                 train_loader, dim_reduction=model_dim, device=device, space='log')

lr_space, loss_space = lr_optim.find_lr()
a = 1