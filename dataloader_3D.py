import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import glob

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}

class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None, ):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)

        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)]
        # print(self.annot)

        self.transform3d = args.ThreeDim

        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes = len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        path_list = np.array([])
        indices = []
        for i, path in enumerate(self.annot['File_Path'].values):
            path_part, _ = os.path.split(path)

            if path_part not in path_list:
                path_list = np.append(path_list, path_part)
                indices = np.append(indices, i)

        self.path_list = path_list
        self._labels = self.annot['Severity_Label'].values[indices.astype(int)]
        assert len(self.path_list) == len(self._labels)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):

        import re

        def natural_key(string_):
            """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        images = glob.glob(self.root + self.path_list[index] + '/*.tif')[:-2]

        if len(images) == 0:
            images = glob.glob(self.root + self.path_list[index] + '/*.png')[:-1]

        images = sorted(images, key=natural_key)

        for i, img_path in enumerate(images):
            img_part, target = Image.open(img_path).convert("L"), self._labels[index]

            if self.transform is not None:
                img_part = self.transform(img_part)

            if i == 0:
                img = img_part
            else:
                img = torch.cat((img, img_part), dim=0)

        # This is kinda gross, but gets the interpolation to function correctly
        img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
        img = torch.nn.functional.interpolate(img, size=(48, 224, 224))
        img = torch.squeeze(img, 0)

        if self.transform3d is not None:
            img = self.transform3d(img)

        return img, target

    def __len__(self):
        return len(self._labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type=str, default='df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type=str, default='df_prime_test.csv')
    parser.add_argument('--data_root', type=str, default='')
    return parser.parse_args()



