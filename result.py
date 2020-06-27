import torch
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import numpy as np


class Result:

    def __init__(self, parameter):
        self.snapshots = []
        self.parameter = parameter
        self.mses = [[]]


    def set_origin(self, batch, labels):
        self.origin_data = batch
        self.origin_labels = labels

    def add_snapshot(self, batch):
        self.snapshots.append(batch)

    def calc_mse(self):
        self.mses = np.zeros((len(self.snapshots),self.parameter["batch_size"]))
        for i_s, s in enumerate(self.snapshots):
            mse = (self.origin_data - s)
            mse = mse ** 2
            mse = mse.sum(3).sum(2)  # sum over 5th and 3rd dimension accumulates pixel diff
            mse = mse / (self.parameter["shape_img"][0] * self.parameter["shape_img"][1])
            self.mses[i_s] = np.squeeze(mse)

    def show(self):
        self.calc_mse()
        fig, subplt = plt.subplots(self.parameter["batch_size"], len(self.snapshots) + 1)
        for i_b in range(self.parameter["batch_size"]):
            # original
            orig = self.origin_data[i_b].view(self.parameter["shape_img"][0], self.parameter["shape_img"][1]).cpu().detach()
            subplt[i_b][0].imshow(orig, cmap='Greys_r')
            subplt[i_b][0].axis('off')
            subplt[i_b][0].title.set_text("Label: {}".format(self.origin_labels.cpu().detach().numpy()[i_b]))

            # recreations
            for i_s, s in enumerate(self.snapshots):
                images_batch = s[i_b].view(self.parameter["shape_img"][0], self.parameter["shape_img"][1]).cpu().detach()
                subplt[i_b][i_s+1].imshow(images_batch, cmap='Greys_r')
                subplt[i_b][i_s + 1].axis('off')
                subplt[i_b][i_s + 1].title.set_text(str(self.mses[i_s][i_b])[:5])

        fig.show()




