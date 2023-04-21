import numpy as np
import torch.nn as nn
import torch.utils.data
import dataloader_3D as dataloader
from torchvision import transforms
import os
import Models
from joblib import dump, load
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import GridSearchCV

if not os.path.isfile(os.getcwd()+'/TrainTest_3D.h5'):

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
        transforms.ToTensor()
    ])

    transform_recon = transforms.Compose([normalize])
    args = dataloader.parse_args()
    args.data_root = os.getcwd()
    trainset = dataloader.OCTDataset(args, 'train', transform=transform)
    testset = dataloader.OCTDataset(args, 'test', transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=3, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=3, shuffle=False)

    model = Models.CAE_3D()
    model.load_state_dict(torch.load(os.getcwd() + '/Models/3D_CAE_Model.pth'))
    model.to(device)


    for iteration, (image, label) in enumerate(train_loader):

        image = normalize(image)
        if iteration == 0:
            image = image.to(device)
            reduced_feat = model.get_representation(image)
            reduced_feat = reduced_feat.to('cpu').detach().numpy()
            X_train = reduced_feat
            y_train = label.numpy()
        else:
            image = image.to(device)
            reduced_feat = model.get_representation(image)
            reduced_feat = reduced_feat.to('cpu').detach().numpy()
            X_train = np.concatenate((X_train, reduced_feat), axis=0)
            y_train = np.concatenate((y_train, label.numpy()))



    for iteration, (image, label) in enumerate(test_loader):
        image = normalize(image)
        if iteration == 0:
            image = image.to(device)
            reduced_feat = model.get_representation(image)
            reduced_feat = reduced_feat.to('cpu').detach().numpy()
            X_test = reduced_feat
            y_test = label.numpy()
        else:
            image = image.to(device)
            reduced_feat = model.get_representation(image)
            reduced_feat = reduced_feat.to('cpu').detach().numpy()
            X_test = np.concatenate((X_test, reduced_feat), axis=0)
            y_test = np.concatenate((y_test, label.numpy()))

    f = h5py.File('TrainTest_3D.h5', 'w')
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('X_test', data=X_test)
    f.create_dataset('y_test', data=y_test)

else:
    f = h5py.File('TrainTest_3D.h5', 'r')
    X_train = f['X_train'][()]
    y_train = f['y_train'][()]
    X_test = f['X_test'][()]
    y_test = f['y_test'][()]

SVM_Classifier = NuSVC()
KNN_Classifier = KNeighborsClassifier()

parameters_svm = {'nu': [0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35], 'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto'], 'class_weight': ['balanced']}
clf = GridSearchCV(SVM_Classifier, parameters_svm, n_jobs=4)
clf.fit(X_train, y_train)
print('Gridsearch Trained')

parameters_knn = {'weights': ['distance'], 'p': [1, 2, 3], 'n_neighbors': [20, 30, 40],
                  'leaf_size': [20, 30, 40]}

clf_knn = GridSearchCV(KNN_Classifier, parameters_knn, n_jobs=4)
clf_knn.fit(X_train, y_train)
knn_train_accuracy = clf_knn.score(X_train, y_train)
knn_test_accuracy = clf_knn.score(X_test, y_test)
print('Train Accuracy: ' + str(knn_train_accuracy) + ' Test Accuracy: ' + str(knn_test_accuracy))

a = 1
