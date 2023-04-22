import numpy as np
import torch.utils.data
import dataloader_3D as dataloader
from torchvision import transforms
import os
import Models
from joblib import dump, load
import torchio as tio
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, classification_report
import h5py
from sklearn.model_selection import GridSearchCV

# This checks weather the sklearn formatted dataset has been created, it makes it if it has not.
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

    spatial = tio.OneOf({
        tio.RandomAffine(): 1,
        tio.RandomElasticDeformation(): 0,
    },
        p=0.75
    )

    weights = torch.Tensor([1 / 159, 1 / 240, 1 / 96])
    ThreeDimTransform = tio.Compose([spatial])
    args = dataloader.parse_args()
    args.data_root = os.getcwd()
    args.ThreeDim = None

    trainset = dataloader.OCTDataset(args, 'train', transform=transform)

    targets = trainset._labels

    reciprocal_weights = np.zeros(len(targets))
    for i, index in enumerate(trainset._labels):
        reciprocal_weights[i] = weights[index]

    sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(reciprocal_weights), len(trainset), replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=8, shuffle=False, sampler=sampler)
    train_loader.dataset.transform3d = None

    testset = dataloader.OCTDataset(args, 'test', transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=3, shuffle=False)

    # Loads convolutional autoencoder
    model = Models.CAE_3D()
    model.load_state_dict(torch.load(os.getcwd() + '/Models/3D_CAE_Model.pth'))
    model.to(device)
    model.eval()

    # Creates train and test dataset in Sklearn usable format

    for epoch in range(1):
        with torch.no_grad():
            for iteration, (image, label) in enumerate(train_loader):

                image = normalize(image)
                if (iteration == 0) and (epoch == 0):
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

    with torch.no_grad():
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

    # Saves data in hdf5 format
    f = h5py.File('TrainTest_3D.h5', 'w')
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('X_test', data=X_test)
    f.create_dataset('y_test', data=y_test)
    f.close()

else:

    # If the data exists already in local dir just load it in
    f = h5py.File('TrainTest_3D.h5', 'r')
    X_train = f['X_train'][()]
    y_train = f['y_train'][()]
    X_test = f['X_test'][()]
    y_test = f['y_test'][()]
    f.close()

# Setup two classifiers
SVM_Classifier = NuSVC()
KNN_Classifier = KNeighborsClassifier()
Train_SVM = True
Train_KNN = True


if Train_SVM:
    # SVM Hyper parameters to perform gridsearch over
    parameters_svm = {'nu': [0.33, 0.34, 0.35, 0.36, 0.374], 'kernel': ['linear', 'rbf'],
                      'gamma': ['scale', 'auto'], 'class_weight': ['balanced']}

    # Fit gridsearch, adjust n jobs to whatever kind of system one is on
    clf = GridSearchCV(SVM_Classifier, parameters_svm, n_jobs=4)
    clf.fit(X_train, y_train)
    y_pred_svm = clf.predict(X_test)

    # Calculate SVM metrics
    test_accuracy_svm = balanced_accuracy_score(y_test, y_pred_svm)
    precision_svm, recall_svm, fbeta_score_svm, support_svm = precision_recall_fscore_support(y_test, y_pred_svm,
                                                                                              labels=np.unique(y_test),
                                                                                              average='weighted')

    print('SVM Classification Report /n')
    print('Balanced Accuracy: ' + str(test_accuracy_svm))
    print(classification_report(y_test, y_pred_svm, target_names=['Class 0', 'Class 1', 'Class 2']))
    print('Total Precision: ' + str(precision_svm))
    print('Total Recall: ' + str(recall_svm))
    print('Total F1 Score: ' + str(fbeta_score_svm))
    dump(clf, os.getcwd() + '/Models/NuSVM_Model.joblib')

if Train_KNN:

    # KNN Hyper parameters to gridsearch over
    parameters_knn = {'weights': ['distance'], 'p': [2], 'n_neighbors': [15, 14],
                      'leaf_size': [18, 17]}

    # Fit KNN gridsearch
    clf_knn = GridSearchCV(KNN_Classifier, parameters_knn, n_jobs=4)
    clf_knn.fit(X_train, y_train)
    y_pred_knn = clf_knn.predict(X_test)

    # Calculate KNN metrics
    test_accuracy_knn = balanced_accuracy_score(y_test, y_pred_knn)
    precision_knn, recall_knn, fbeta_score_knn, support_knn = precision_recall_fscore_support(y_test, y_pred_knn,
                                                                                              labels=np.unique(y_test),
                                                                                              average='weighted')
    print('KNN Classification Report /n')
    print('Balanced Accuracy: ' + str(test_accuracy_knn))
    print(classification_report(y_test, y_pred_knn, target_names=['Class 0', 'Class 1', 'Class 2']))
    print('Total Precision: ' + str(precision_knn))
    print('Total Recall: ' + str(recall_knn))
    print('Total F1 Score: ' + str(fbeta_score_knn))

    dump(clf, os.getcwd() + '/Models/KNN_Model.joblib')


