import torch.nn as nn
import torch.utils.data
import dataloader
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
from sklearn.metrics import balanced_accuracy_score
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


transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    #transforms.RandomCrop((224,224)),
    #transforms.RandomRotation(10),
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
testset = dataloader.OCTDataset(args, 'test', transform=transform_test)

#class_dist = 1 / np.array([7788, 11760, 4704])




test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=True)

weights = torch.Tensor([1.03801, 0.68741, 1.71854])

model = torchvision.models.GoogLeNet()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=1024, out_features=3, bias=True)
model.aux1.fc2 = nn.Linear(in_features=1024, out_features=3, bias=True)
model.aux2.fc2 = nn.Linear(in_features=1024, out_features=3, bias=True)

pretrained = True
if pretrained:
    model.load_state_dict(torch.load(os.getcwd() + '/CNN_Model.pth'))

model.to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=weights.to(device), reduction='mean')
#optimizer = torch.optim.Adam(model.parameters(),
#                             lr=1e-2,
#                             weight_decay=1e-6)

optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-2,
                            momentum=0.9,
                            nesterov=True)
#optimizer = torch.optim.NAdam(model.parameters(),
#                              lr=1e-5,
#                              weight_decay=1e-6)

metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)
f1_score = torchmetrics.F1Score(task='multiclass', num_classes=3)
precision = torchmetrics.Precision(task='multiclass', num_classes=3)
recall = torchmetrics.Recall(task='multiclass', num_classes=3)

epochs = 2500
outputs = []
total_loss = []
train = False

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
#                                                steps_per_epoch=len(train_loader), epochs=epochs)

if train:

    trainset = dataloader.OCTDataset(args, 'train', transform=transform)

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

        model.train()
        for iteration_train, (image_train, label_train) in enumerate(train_loader):

            optimizer.zero_grad()
            image_train = image_train.to(device)
            pred_train = model(image_train)
            loss_train = loss_function(pred_train[0], label_train.to(device)) + \
                         0.3*loss_function(pred_train[1], label_train.to(device)) + \
                         0.3*loss_function(pred_train[2], label_train.to(device))
            loss_train.backward()
            optimizer.step()
            #scheduler.step()

            train_loss += loss_train.to('cpu').detach().numpy()

            pred_label_train = torch.argmax(pred_train[0], dim=1)
            train_accuracy = metric(pred_train[0].to('cpu').detach(), label_train)

        train_accuracy = metric.compute().numpy()
        metric.reset()

        model.eval()
        y_true = np.array([])
        y_pred = np.array([])
        for iteration_val, (image_val, label_val) in enumerate(val_loader):

            with torch.no_grad():
                image_val = image_val.to(device)
                pred_val = model(image_val)
                loss_val = loss_function(pred_val, label_val.to(device))
                val_loss += loss_val.to('cpu').detach().numpy()

                pred_label_val = torch.argmax(pred_val, dim=1)

                val_accuracy = metric(pred_val.to('cpu').detach(), label_val)
                #val_f1 = f1_score(pred_val.to('cpu').detach(), label_val)
                y_true = np.concatenate((y_true, label_val))
                y_pred = np.concatenate((y_pred, pred_label_val.cpu().detach().numpy()))


        val_accuracy = balanced_accuracy_score(y_true, y_pred)
        l1_acc = ((y_true == 1) & (y_pred == 1)).sum() / np.sum(y_true == 1)
        l0_acc = ((y_true == 0) & (y_pred == 0)).sum() / np.sum(y_true == 0)
        l2_acc = ((y_true == 2) & (y_pred == 2)).sum() / np.sum(y_true == 2)
        metric.reset()


        print('Epoch ' + str(epoch + 1) + ' --- Train Loss: ' + str(train_loss) + ' --- Val Loss: ' + str(val_loss) +
              ' --- Train Accuracy: ' + str(train_accuracy) + ' --- Val Accuracy: ' + str(val_accuracy))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.getcwd() + '/CNN_Model.pth')

test_loss = 0
test_acc = []
f1_test = 0

model.eval()
test_accuracy = 0
y_true = np.array([])
y_pred = np.array([])
for iteration_test, (image_test, label_test) in enumerate(test_loader):

    with torch.no_grad():
        image_test = image_test.to(device)
        pred_test = model(image_test)
        loss_test = loss_function(pred_test, label_test.to(device))
        test_loss += loss_test.to('cpu').detach().numpy()

        pred_label_test = torch.argmax(pred_test, dim=1)

        y_true = np.concatenate((y_true, label_test))
        y_pred = np.concatenate((y_pred, pred_label_test.cpu().detach().numpy()))



test_accuracy = balanced_accuracy_score(y_true, y_pred)
l1_acc = ((y_true == 1) & (y_pred==1)).sum() / np.sum(y_true == 1)
l0_acc = ((y_true == 0) & (y_pred==0)).sum() / np.sum(y_true == 0)
l2_acc = ((y_true == 2) & (y_pred==2)).sum() / np.sum(y_true == 2)


print('Test Loss: ' + str(test_loss) + ' --- Test Accuracy: ' + str(test_accuracy) + ' --- Test F1 Score: ' + str(f1_test))
metric.reset()