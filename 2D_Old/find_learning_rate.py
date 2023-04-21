import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import h5py

class find_learning_rate():

    def __init__(self, min_lr, max_lr, num_points, epochs, model, optimizer, loss_fn, data, dim_reduction=None,
                 device='cpu', space='linear'):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs
        self.num_points = num_points
        self.final_loss = []
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data = data
        self.dim_reduction = dim_reduction
        self.device = device
        self.lr_space = []
        self.space = space

    def get_lr_space(self):

        if self.space == 'linear':
            lr_space = np.linspace(self.min_lr, self.max_lr, self.num_points)
        elif self.space == 'log':
            lr_space = np.logspace(np.log10(self.min_lr), np.log10(self.max_lr), self.num_points)

        self.lr_space = lr_space
        return lr_space

    def train_lr(self):

        for lr in self.get_lr_space():

            self.optimizer.defaults['lr'] = lr

            for epoch in np.arange(self.epochs):

                tot_loss = 0

                for _, (x, y) in enumerate(self.data):

                    if self.dim_reduction is not None:

                        with torch.no_grad():
                            x = x.to(self.device)
                            x = self.dim_reduction.get_representation(x)

                    self.optimizer.zero_grad()

                    pred_y = self.model(x)
                    loss = self.loss_fn(pred_y, y.to(self.device))
                    loss.backward()
                    self.optimizer.step()

                    tot_loss += loss.to('cpu').detach().numpy()

                print('Completed epoch ' + str(epoch) + ' for lr ' + str(lr) + ' loss is ' + str(tot_loss.item()))

            self.final_loss.append(tot_loss.item())

    def find_lr(self):

        self.train_lr()

        plt.figure()
        plt.semilogx(self.lr_space, self.final_loss)
        plt.show()
        plt.savefig('LearningRate.jpg')

        return self.lr_space, self.final_loss



