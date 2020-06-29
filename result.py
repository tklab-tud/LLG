import matplotlib.pyplot as plt
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
        self.mses = np.zeros((len(self.snapshots), self.parameter["batch_size"]))
        for i_s, s in enumerate(self.snapshots):
            self.mses[i_s] = np.squeeze(self.mse(self.origin_data, s))

    def mse(self, a, b):
        mse = (a - b)
        mse = mse ** 2
        mse = mse.sum(len(a.size()) - 1).sum(len(a.size()) - 2)  # sum over the last 2 dimension accumulates pixel diff
        mse = mse / (self.parameter["shape_img"][0] * self.parameter["shape_img"][1])
        return mse

    def swap_samples_in_snapshots(self, a, b):
        for snap in self.snapshots:
            tmp = snap[a].clone()
            snap[a] = snap[b]
            snap[b] = tmp

    def fix_snapshot_order(self):
        for i_o, orig in enumerate(self.origin_data):
            err = []
            for i_r in range(self.parameter["batch_size"]):
                recreation = self.snapshots[-1][i_r]
                err.append(self.mse(orig, recreation))

            if np.argmin(err) != i_o:
                self.swap_samples_in_snapshots(np.argmin(err), i_o)

    def show(self):
        self.fix_snapshot_order()
        self.calc_mse()

        fig, subplots = plt.subplots(self.parameter["batch_size"], len(self.snapshots) + 1)

        # fix subplots returning obj instead of array at bs = 1
        if self.parameter["batch_size"]:
            subplots = [subplots]

        fig.set_size_inches((len(self.snapshots) + 1) * self.parameter["shape_img"][0] / 10,
                            len(self.origin_data) * self.parameter["shape_img"][1] / 10)

        for i_b in range(self.parameter["batch_size"]):
            # original
            orig = self.origin_data[i_b].view(self.parameter["shape_img"][0],
                                              self.parameter["shape_img"][1]).cpu().detach()
            subplots[i_b][0].imshow(orig, cmap='Greys_r')
            subplots[i_b][0].axis('off')
            subplots[i_b][0].title.set_text("Label: {}".format(self.origin_labels.cpu().detach().numpy()[i_b]))

            # recreations
            for i_s, s in enumerate(self.snapshots):
                images_batch = s[i_b].view(self.parameter["shape_img"][0],
                                           self.parameter["shape_img"][1]).cpu().detach()
                subplots[i_b][i_s + 1].imshow(images_batch, cmap='Greys_r')
                subplots[i_b][i_s + 1].axis('off')
                subplots[i_b][i_s + 1].title.set_text(str(self.mses[i_s][i_b])[:7])

        fig.show()
