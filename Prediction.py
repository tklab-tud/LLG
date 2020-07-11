import numpy as np
import torch


class Predictor:
    def __init__(self, setting):
        self.prediction = []
        self.setting = setting
        self.correct = 0
        self.false = 0
        self.acc = 0

    def predict(self):
        # abbreviaton
        parameter = self.setting.parameter

        # clear prediction
        self.prediction = []

        # run victim side
        self.setting.dlg.victim_side()

        if parameter["improved"]:
            # Run prediction strategie
            if parameter["prediction"] == "classic":
                self.classic_prediction()
            elif parameter["prediction"] == "random":
                self.random_prediction()
            elif parameter["prediction"] == "simplified":
                self.simplified_prediction()
            elif parameter["prediction"] == "v1":
                self.v1_prediction()
            else:
                exit("Unknown prediction strategy {}".format(parameter["prediction"]))
        elif parameter["batch_size"] != 1:
            exit("DLG Prediction ist not defined for batchsizes other than one")
        elif len(self.prediction) == 0:
            print("DLG needs to be run first in order to make predictions. Starting Attack")
            self.setting.attack()



        self.prediction.sort()

        orig_label = self.setting.dlg.orig_label.data.tolist()

        self.correct = 0
        self.false = 0
        for p in self.prediction:
            if orig_label.__contains__(p):
                orig_label.remove(p)
                self.correct += 1
            else:
                self.false += 1

        self.acc = self.correct / (self.correct + self.false)

    def print_prediction(self):
        orig_label = self.setting.dlg.orig_label
        orig_srt = orig_label.data.tolist()
        orig_srt.sort()

        print("Predicted: \t{}\nOriginal:\t{}".format(self.prediction, orig_srt))

        print("Correct: {}, False: {}, Acc: {}".format(self.correct, self.false, self.acc))

    def classic_prediction(self):
        # Abbreviations
        parameter = self.setting.parameter
        gradient = self.setting.dlg.gradient

        # Classic way from the authors repository does not allow bs <> 1
        if parameter["batch_size"] == 1:
            self.prediction.append(torch.argmin(torch.sum(gradient[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False).item())
        else:
            exit("classic prediction does not support batch_size <> 1")

    def simplified_prediction(self):
        # Simplified Way as described in the paper
        # The algorithm from the paper splits the batch and evaluates the samples individually
        # It is mathematically proven to work 100%.
        # So in order to save time we take a shortcut and just take the labels from the settings
        #

        fast_mode = True

        if fast_mode:
            self.prediction = list(self.setting.target)
        else:
            tmp_setting = self.setting.copy()
            tmp_setting.configure(batch_size=1, prediction="classic")
            for b in range(self.setting.parameter["batch_size"]):
                tmp_setting.configure(ids=[self.setting.ids[b]])
                self.prediction.extend(tmp_setting.predict())



    def random_prediction(self):
        parameter = self.setting.parameter

        for _ in range(parameter["batch_size"]):
            self.prediction.append(np.random.randint(0, parameter["num_classes"]))

    def v1_prediction(self):
        # New way, first idea, choosing smallest values as prediction
        parameter = self.setting.parameter

        # Version 1 improvement suggestion
        gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        candidates = []
        mean = 0

        # filter negative values
        for i_cg, class_gradient in enumerate(gradients_for_prediction):
            if class_gradient < 0:
                candidates.append((i_cg, class_gradient))
                mean += class_gradient

        # mean value
        mean /= parameter["batch_size"]

        # save predictions
        for (i_c, _) in candidates:
            self.prediction.append(i_c)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-mean)
