import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class Result:

    def __init__(self, parameter):
        self.snapshots = []
        self.parameter = parameter
        self.mses = [[]]
        self.losses = []

    def set_origin(self, batch, labels):
        self.origin_data = batch
        self.origin_labels = labels

    def add_snapshot(self, batch):
        self.snapshots.append(batch)

    def add_loss(self, loss):
        self.losses.append(loss)

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

    def realign_snapshops(self, alignment):
        #make a temporary copy
        tmp = [x.clone() for x in self.snapshots]

        # fix one row after another to be correct aligned
        for i_orig, i_reco in enumerate(alignment):
            # fix the row by going through every snapshot at the position of the row
            for i_s, snap in enumerate(self.snapshots):
                snap[i_orig] = tmp[i_s][i_reco]



    def fix_snapshot_order(self):
        #fill the mse matrix
        err = [[1 for x in range(self.parameter["batch_size"])] for x in range(self.parameter["batch_size"])]
        for i_o, orig in enumerate(self.origin_data):
            for i_b in range(self.parameter["batch_size"]):
                err[i_o][i_b] = torch.sum(self.mse(orig, self.snapshots[-1][i_b]))

        alignment = [1 for x in range(self.parameter["batch_size"])]

        for _ in range(self.parameter["batch_size"]):
            # find best alignment
            max_orig = torch.Tensor(err).argmin() // self.parameter["batch_size"]
            max_reco = torch.Tensor(err).argmin() % self.parameter["batch_size"]
            max_reco = max_reco.data.item()
            alignment[max_orig] = max_reco

            # purge column
            for orig in range(self.parameter["batch_size"]):
                err[orig][max_reco] = torch.Tensor([1])

            # purge row
            for reco in range(self.parameter["batch_size"]):
                err[max_orig][reco] = torch.Tensor([1])

        self.realign_snapshops(alignment)


    def show(self):
        self.fix_snapshot_order()
        self.calc_mse()

        fig, subplots = plt.subplots(self.parameter["batch_size"], len(self.snapshots) + 1)

        # fix subplots returning obj instead of array at bs = 1
        if self.parameter["batch_size"] == 1:
            subplots = [subplots]

        fig.set_size_inches((len(self.snapshots) + 1) * self.parameter["shape_img"][0] / 10,
                            len(self.origin_data) * self.parameter["shape_img"][1] / 10)

        for i_b in range(self.parameter["batch_size"]):
            # original
            orig = np.squeeze(self.origin_data[i_b].numpy())

            if self.parameter["channel"] == 1:
                subplots[i_b][0].imshow(orig, cmap="Greys_r")
            elif self.parameter["channel"] == 3:
                rgb_img = cv2.merge([orig[0], orig[1], orig[2]])
                subplots[i_b][0].imshow(rgb_img)

            subplots[i_b][0].axis('off')
            subplots[i_b][0].title.set_text("Label: {}".format(self.origin_labels.cpu().detach().numpy()[i_b]))

            # recreations
            for i_s, s in enumerate(self.snapshots):
                images_batch = np.squeeze(s[i_b].numpy())

                if self.parameter["channel"] == 1:
                    subplots[i_b][i_s + 1].imshow(images_batch, cmap="Greys_r")
                elif self.parameter["channel"] == 3:
                    rgb_img = cv2.merge([images_batch[0], images_batch[1], images_batch[2]])
                    subplots[i_b][i_s + 1].imshow(rgb_img)

                subplots[i_b][i_s + 1].axis('off')
                subplots[i_b][i_s + 1].title.set_text(
                    "mse:{:.8f}\nloss:{:.8f}".format(self.mses[i_s][i_b], self.losses[i_s].item()))

        fig.show()
