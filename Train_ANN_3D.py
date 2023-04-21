import numpy as np
import torch.utils.data
import dataloader_3D as dataloader
from torchvision import transforms
import os
import Models
import matplotlib.pyplot as plt
import torchmetrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import torchio as tio

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

weights = torch.Tensor([1/7788, 1/11760, 1/4704])

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

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=3, shuffle=True)

model = Models.ClassifierANN(18816, 3, 2500, 3)
model_dim = Models.CAE_3D()
model_dim.load_state_dict(torch.load(os.getcwd() + '/Models/3D_CAE_Model.pth'))

pretrained = False

if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/Models/ANN_3D.pth'))

model.to(device)
model_dim.to(device)

loss_function = torch.nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(),
#                            lr=1e-3,
#                            momentum=0.9,
#                            nesterov=True,
#                            weight_decay=1e-6)

optimizer = torch.optim.Adam(model.parameters(),
                              lr=1e-4,
                              weight_decay=1e-6)

metric = torchmetrics.Accuracy(task='multiclass', num_classes=3, average='macro', multidim_average='global')

epochs = 100
outputs = []
total_loss = []
train = True
test = False

if train:

    trainset = dataloader.OCTDataset(args, 'train', transform=transform)

    targets = trainset._labels

    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

    train_sub = torch.utils.data.Subset(trainset, train_idx)
    val_sub = torch.utils.data.Subset(trainset, val_idx)

    reciprocal_weights = np.zeros(len(train_sub))

    for i, index in enumerate(trainset._labels[train_sub.indices]):
        reciprocal_weights[i] = weights[index]

    sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(reciprocal_weights), len(train_sub), replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_sub, batch_size=4, shuffle=False, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=3, shuffle=False)

    train_loss_hist = []
    val_loss_hist = []
    train_accuracy_hist = []
    val_accuracy_hist = []

    for epoch in range(epochs):

        train_loss = 0
        test_loss = 0
        val_loss = 0

        model.train()
        for iteration_train, (image_train, label_train) in enumerate(train_loader):

            with torch.no_grad():
                image_train = normalize(image_train)
                image_train = image_train.to(device)
                IM_cae_train = model_dim.get_representation(image_train)

            optimizer.zero_grad()

            pred_train = model(IM_cae_train)
            loss_train = loss_function(pred_train, label_train.to(device))
            loss_train.backward()
            optimizer.step()

            train_loss += loss_train.to('cpu').detach().numpy()

            pred_label_train = torch.argmax(pred_train, dim=1)
            train_accuracy = metric(pred_train.to('cpu').detach(), label_train)

        train_accuracy = metric.compute().numpy()
        metric.reset()

        model.eval()

        for iteration_val, (image_val, label_val) in enumerate(val_loader):

            with torch.no_grad():
                image_val = normalize(image_val)
                image_val = image_val.to(device)
                IM_CAE_val = model_dim.get_representation(image_val)
                pred_val = model(IM_CAE_val)
                loss_val = loss_function(pred_val, label_val.to(device))
                val_loss += loss_val.to('cpu').detach().numpy()

                pred_label_test = torch.argmax(pred_val, dim=1)
                val_accuracy = metric(pred_val.to('cpu').detach(), label_val)

        val_accuracy = metric.compute().numpy()
        metric.reset()

        print('Epoch ' + str(epoch + 1) + ' --- Train Loss: ' + str(train_loss) + ' --- Val Loss: ' + str(val_loss) +
              ' --- Train Accuracy: ' + str(train_accuracy) + ' --- Val Accuracy: ' + str(val_accuracy))

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_accuracy)
        train_accuracy_hist.append(train_accuracy)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.getcwd() + '/Models/ANN_3D.pth')

    train_loss_hist = np.array(train_loss_hist)
    train_accuracy_hist = np.array(train_accuracy_hist)
    val_loss_hist = np.array(val_loss_hist)
    val_accuracy_hist = np.array(val_accuracy_hist)

    train_loss_hist /= train_loss_hist[0]
    val_loss_hist /= val_loss_hist[0]

    plt.figure()
    plt.plot(range(epochs), train_loss_hist, 'r')
    plt.plot(range(epochs), val_loss_hist, 'b')
    plt.plot(range(epochs), train_accuracy_hist, 'r--')
    plt.plot(range(epochs), val_accuracy_hist, 'b--')
    plt.legend(('Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('ANN Accuracy and Loss')
    plt.savefig(os.getcwd() + '/Plots/ANN_3D_Loss.png')


if test:
    test_loss = 0
    test_acc = []
    f1_test = 0
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    for iteration_test, (image_test, label_test) in enumerate(test_loader):

        with torch.no_grad():
            image_test = normalize(image_test)
            image_test = image_test.to(device)
            IM_CAE_Test = model_dim.get_representation(image_test)
            pred_test = model(IM_CAE_Test)
            loss_test = loss_function(pred_test, label_test.to(device))
            test_loss += loss_test.to('cpu').detach().numpy()

            pred_label_test = torch.argmax(pred_test, dim=1)
            test_accuracy = metric(pred_test.to('cpu').detach(), label_test)
            y_true = np.concatenate((y_true, label_test))
            y_pred = np.concatenate((y_pred, pred_label_test.cpu().detach().numpy()))


    test_accuracy = metric.compute().numpy()
    val_accuracy = balanced_accuracy_score(y_true, y_pred)
    l1_acc = ((y_true == 1) & (y_pred == 1)).sum() / np.sum(y_true == 1)
    l0_acc = ((y_true == 0) & (y_pred == 0)).sum() / np.sum(y_true == 0)
    l2_acc = ((y_true == 2) & (y_pred == 2)).sum() / np.sum(y_true == 2)

    print('Test Loss: ' + str(test_loss) + ' --- Test Accuracy: ' + str(test_accuracy))
    metric.reset()



