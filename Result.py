import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Result:

    def __init__(self, setting):
        self.setting = setting
        self.parameter = setting.parameter
        self.snapshots = []
        self.mses = np.array([])
        self.losses = []
        self.composed_fig = None
        self.composed_subplots = None
        self.separate_figs = []
        self.unprocessed = True
        self.origin_data = None
        self.origin_labels = None
        plt.rcParams.update({'figure.max_open_warning': 0})

    def set_origin(self, batch, labels):
        self.origin_data = batch
        self.origin_labels = labels
        self.unprocessed = True

    def add_snapshot(self, batch):
        self.snapshots.append(batch)
        self.unprocessed = True

    def add_loss(self, loss):
        self.losses.append(loss)
        self.unprocessed = True

    def calc_mse(self):
        self.mses = np.zeros((len(self.snapshots), self.setting.parameter["batch_size"]))
        for i_s, s in enumerate(self.snapshots):
            self.mses[i_s] = self.mse(self.origin_data, s).sum(1)

    def mse(self, a, b):
        mse = (a - b)
        mse = mse ** 2
        mse = mse.sum(-1).sum(-2)  # sum over the last 2 dimension accumulates pixel diff
        mse = mse / (self.parameter["shape_img"][0] * self.parameter["shape_img"][1])
        return mse

    def realign_snapshops(self, alignment):
        # make a temporary copy
        tmp = [x.copy() for x in self.snapshots]

        # fix one row after another to be correct aligned
        for i_orig, i_reco in enumerate(alignment):
            # fix the row by going through every snapshot at the position of the row
            for i_s, snap in enumerate(self.snapshots):
                snap[i_orig] = tmp[i_s][i_reco]

    def fix_snapshot_order(self):
        # fill the mse matrix
        err = [[1 for x in range(self.parameter["batch_size"])] for x in range(self.parameter["batch_size"])]
        for i_o, orig in enumerate(self.origin_data):
            for i_b in range(self.parameter["batch_size"]):
                err[i_o][i_b] = np.sum(self.mse(orig, self.snapshots[-1][i_b]))

        alignment = [1 for x in range(self.parameter["batch_size"])]

        for _ in range(self.parameter["batch_size"]):
            # find best alignment
            max_orig = np.argmin(err) // self.parameter["batch_size"]
            max_reco = np.argmin(err) % self.parameter["batch_size"]
            alignment[max_orig] = max_reco

            # purge column
            for orig in range(self.parameter["batch_size"]):
                err[orig][max_reco] = [1]

            # purge row
            for reco in range(self.parameter["batch_size"]):
                err[max_orig][reco] = [1]

        self.realign_snapshops(alignment)

    def add_seperate_image(self, item, original, batch, snap):
        item = np.squeeze(item)
        fig, subplot = plt.subplots(1, 1)

        if self.parameter["channel"] == 1:
            subplot.imshow(np.squeeze(item), cmap="Greys_r")
        elif self.parameter["channel"] == 3:
            rgb_img = cv2.merge([item[0], item[1], item[2]])
            subplot.imshow(rgb_img)

        self.separate_figs.append((fig, batch, snap, original))

    def add_composed_image(self, item, original, batch, snap):

        row = batch
        if original:
            column = 0
        else:
            column = snap + 1

        subplot = self.composed_subplots[row][column]

        if self.parameter["channel"] == 1:
            subplot.imshow(np.squeeze(item), cmap="Greys_r")
        elif self.parameter["channel"] == 3:
            rgb_img = cv2.merge([item[0], item[1], item[2]])
            subplot.imshow(rgb_img)

        subplot.axis('off')

        if original:
            subplot.title.set_text("Label: {}".format(self.origin_labels[batch]))
        else:
            subplot.title.set_text("mse:{:.8f}\nloss:{:.8f}".format(self.mses[snap][batch], self.losses[snap]))

    def update_figures(self):
        if self.unprocessed:
            self.fix_snapshot_order()
            self.calc_mse()

            # initialise composed fig
            self.composed_fig, self.composed_subplots = plt.subplots(self.parameter["batch_size"],
                                                                     len(self.snapshots) + 1)
            self.composed_fig.set_size_inches((len(self.snapshots) + 1) * self.parameter["shape_img"][0] / 10,
                                              len(self.origin_data) * self.parameter["shape_img"][1] / 10)

            # fix subplots returning obj instead of array at bs = 1
            if self.parameter["batch_size"] == 1:
                self.composed_subplots = [self.composed_subplots]

            # Generate Images
            # Iterate over Originals
            for i_b, orig in enumerate(self.origin_data):
                # Original Image
                self.add_composed_image(orig, True, i_b, None)
                self.add_seperate_image(orig, True, i_b, None)

                # Recreations
                for i_s, snap in enumerate(self.snapshots):
                    self.add_composed_image(snap[i_b], False, i_b, i_s)
                    self.add_seperate_image(snap[i_b], False, i_b, i_s)

            self.unprocessed = False

    def show_composed_image(self):
        if self.composed_fig is not None:
            self.update_figures()
            self.composed_fig.show()

    def store_composed_image(self):
        if self.composed_fig is not None:
            self.update_figures()
            if not os.path.exists(self.parameter["result_path"]):
                os.makedirs(self.parameter["result_path"])

            self.composed_fig.savefig(
                self.parameter["result_path"] + "composed_image{}.png".format(self.parameter["run_name"]))

    def store_separate_images(self):
        if self.composed_fig is not None:
            self.update_figures()
        if not os.path.exists(self.parameter["result_path"] + "Images{}".format(self.parameter["run_name"])):
            os.makedirs(self.parameter["result_path"] + "Images{}".format(self.parameter["run_name"]))

        for (fig, batch, snap, original) in self.separate_figs:
            if original:
                fig.savefig(
                    self.parameter["result_path"] + "Images{}/{:03d}_original.png".format(self.parameter["run_name"],
                                                                                          batch))
            else:
                fig.savefig(
                    self.parameter["result_path"] + "Images{}/{:03d}s{:04d}.png".format(self.parameter["run_name"],
                                                                                        batch, snap))

    def store_data(self):
        if self.composed_fig is not None:
            self.update_figures()

        if not os.path.exists(self.parameter["result_path"]):
            os.makedirs(self.parameter["result_path"])

        # fill dictionary with parameter, data and prediction

        data_dic = {
            "parameter": self.parameter,
            "losses": self.losses,
            "mses": self.mses.tolist(),
            "prediction": {
                "correct": self.setting.predictor.correct,
                "false": self.setting.predictor.false,
                "accuracy": self.setting.predictor.acc,
                "prediction": self.setting.predictor.prediction,
            },
            "target": self.setting.target,
            "ids":self.setting.ids, #[x.item() for x in self.setting.ids],
            "snapshots": list(map(lambda x: x.tolist(), self.snapshots))
        }

        # dump to json
        with open(self.parameter["result_path"] + "data{}.json".format(self.parameter["run_name"]), "w") as file:
            json.dump(data_dic, file)

    def delete(self):
        plt.close(self.composed_fig)
        for fig, _, _, _ in self.separate_figs:
            plt.close(fig)

    def store_everything(self):
        self.store_composed_image()
        self.store_separate_images()
        self.store_data()






