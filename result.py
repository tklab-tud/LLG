import torch
import matplotlib.pyplot as plt


class Result:

    def __init__(self, parameter):
        self.snapshots = []
        self.mses = []
        self.parameter = parameter

    def set_origin(self, batch, labels):
        self.origin_data = batch
        self.origin_labels = labels

    def add_snapshot(self, batch):
        self.snapshots.append(batch)
        mse = self.origin_data - batch
        mse = mse ** 2
        mse = mse.sum(3)  # sum over 3rd dimension, shouldn't mix up batches
        mse = mse/ (self.parameter["shape_img"][0]*self.parameter["shape_img"][1])
        self.mses.append(mse)

    def show(self):
        for b in range(self.parameter["batch_size"]):
            for i_s, s in enumerate(self.snapshots, 1):
                plt.subplot(self.parameter["batch_size"], len(self.snapshots), b*len(self.snapshots)+i_s)
                images_batch = s[b].view(self.parameter["shape_img"][0], self.parameter["shape_img"][1]).cpu().detach()
                plt.imshow(images_batch, cmap='Greys_r')
                plt.axis('off')

        plt.show()




