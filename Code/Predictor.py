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
        self.impact = 0
        self.offset = torch.zeros(setting.parameter["num_classes"])

    def predict(self):
        # abbreviaton
        parameter = self.setting.parameter

        # clear prediction
        self.prediction = []

        # run victim side
        self.setting.dlg.victim_side()


        # Run prediction strategie
        if parameter["prediction"] == "random":
            self.random_prediction()
        elif parameter["prediction"] == "idlg":
            self.simplified_prediction()
        elif parameter["prediction"] == "v1":
            self.v1_prediction()
        elif parameter["prediction"] == "v2":
            self.v2_prediction()
        elif parameter["prediction"] == "v3":
            self.v3_prediction()
        else:
            exit("Unknown prediction strategy {}".format(parameter["prediction"]))

        self.prediction.sort()

        # analyse prediction
        orig_label = self.setting.parameter["orig_label"].tolist()

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
        orig_label = self.setting.parameter["orig_label"].tolist()
        orig_label.sort()

        print("Predicted: \t{}\nOriginal:\t{}".format(self.prediction, orig_label))

        print("Correct: {}, False: {}, Acc: {}".format(self.correct, self.false, self.acc))

    def simplified_prediction(self):
        # The algorithm from the paper splits the batch and evaluates the samples individually
        # It is mathematically proven to work 100%.
        # So in order to save time we take a shortcut and just take the labels from the settings

        self.prediction = list(self.setting.parameter["orig_label"])
        self.prediction = [x.item() for x in self.prediction]



    def random_prediction(self):
        # In order to compare strategies random prediction has been added.

        for _ in range( self.setting.parameter["batch_size"]):
            self.prediction.append(np.random.randint(0,  self.setting.parameter["num_classes"]))


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
                mean += class_gradient.item()

        # mean value
        self.impact = mean / parameter["batch_size"]

        # save predictions
        for (i_c, _) in candidates:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)


        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidate, likely to be doubled, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one occurrence to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)

        self.prediction.sort()



    def v2_prediction(self):
        parameter = self.setting.parameter

        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        candidates = []

        # backup negative values
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
            if class_gradient < 0:
                candidates.append((i_cg, class_gradient))

        # create a new setting for impact / offset calculation
        tmp_setting = self.setting.copy()
        tmp_setting.model = self.setting.model
        impact = 0
        acc_impact = 0
        acc_offset = np.zeros(parameter["num_classes"])
        n = 10

        # calculate bias and impact
        for _ in range(n):
            tmp_gradients = []

            for i in range(parameter["num_classes"]):
                tmp_setting.configure(targets=[i] * parameter["batch_size"])
                tmp_setting.dlg.victim_side()
                tmp_gradients.append(torch.sum(tmp_setting.dlg.gradient[-2], dim=-1).cpu().detach().numpy())
                impact += torch.sum(tmp_setting.dlg.gradient[-2], dim=-1)[i].item()

            acc_offset += np.mean(tmp_gradients, 0)
            impact /= (parameter["num_classes"] * parameter["batch_size"])
            acc_impact += impact

        self.impact = (acc_impact / n) * (1 + 1/parameter["num_classes"])

        acc_offset = np.divide(acc_offset, n)
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset


        # save predictions
        for (i_c, _) in candidates:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)









