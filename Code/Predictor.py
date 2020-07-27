import numpy as np
import torch


class Predictor:
    def __init__(self, setting):
        self.prediction = []
        self.setting = setting
        self.correct = 0
        self.false = 0
        self.acc = 0
        self.gradients_for_prediction = []

    def predict(self):
        # abbreviaton
        parameter = self.setting.parameter

        # clear prediction
        self.prediction = []

        # run victim side
        self.setting.dlg.victim_side()

        if parameter["improved"]:
            # Run prediction strategie
            if parameter["prediction"] == "random":
                self.random_prediction()
            elif parameter["prediction"] == "idlg":
                self.simplified_prediction()
            elif parameter["prediction"] == "v1":
                self.v1_prediction()
            elif parameter["prediction"] == "v2":
                self.v2_prediction()
            else:
                exit("Unknown prediction strategy {}".format(parameter["prediction"]))
        elif parameter["batch_size"] != 1:
            exit("DLG Prediction ist not defined for batch sizes other than one")
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
        print(self.setting.parameter["prediction"], ": ACC: ", self.acc)

    def print_prediction(self):
        orig_label = self.setting.dlg.orig_label
        orig_srt = orig_label.data.tolist()
        orig_srt.sort()

        print("Predicted: \t{}\nOriginal:\t{}".format(self.prediction, orig_srt))

        print("Correct: {}, False: {}, Acc: {}".format(self.correct, self.false, self.acc))



    def simplified_prediction(self):
        # Simplified Way as described in the paper
        # The algorithm from the paper splits the batch and evaluates the samples individually
        # It is mathematically proven to work 100%.
        # So in order to save time we take a shortcut and just take the labels from the settings
        #

        fast_mode = True

        if fast_mode:
            self.prediction = list(self.setting.target[:self.setting.parameter["batch_size"]])
        else:
            tmp_setting = self.setting.copy()
            tmp_setting.configure(batch_size=1, prediction="classic")
            for b in range(self.setting.parameter["batch_size"]):
                tmp_setting.configure(ids=[self.setting.ids[b]])
                self.prediction.extend(tmp_setting.predict())

        self.prediction.sort()

    def random_prediction(self):
        parameter = self.setting.parameter

        for _ in range(parameter["batch_size"]):
            self.prediction.append(np.random.randint(0, parameter["num_classes"]))

        self.prediction.sort()

    def v1_prediction(self):
        # New way, first idea, choosing smallest values as prediction
        parameter = self.setting.parameter

        # Version 1 improvement suggestion
        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        candidates = []
        mean = 0


        # filter negative values
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
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
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-mean)

        self.prediction.sort()

    def v2_prediction(self):
        # Version 2 includes Gradient-Substraction
        parameter = self.setting.parameter

        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        candidates = []

        # backup negative values
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
            if class_gradient < 0:
                candidates.append((i_cg, class_gradient))

        # NetBias
        netbias, mean = self.get_netbias()
        self.gradients_for_prediction -= netbias

        # save predictions
        for (i_c, _) in candidates:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-mean)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-mean)

        self.prediction.sort()

    def get_netbias(self):
        parameter = self.setting.parameter
        # create a new setting
        tmp_setting = self.setting.copy()
        tmp_setting.model = self.setting.model
        tmp_gradients = []
        impact = 0

        # calculate bias and impact
        for i in range(parameter["num_classes"]):
            tmp_setting.configure(target=[i]*parameter["batch_size"])
            tmp_setting.dlg.victim_side()
            tmp_gradients.append(torch.sum(tmp_setting.dlg.gradient[-2], dim=-1).cpu().detach().numpy())
            impact += torch.sum(tmp_setting.dlg.gradient[-2], dim=-1)[i].item()

        bias = torch.Tensor(np.mean(tmp_gradients, 0)).to(self.setting.device)
        impact /= (parameter["num_classes"] * parameter["batch_size"])
        return bias, impact * 1.11





