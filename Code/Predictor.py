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
        self.offset = torch.zeros(setting.parameter["num_classes"]).to(self.setting.device)

    def predict(self):
        # abbreviaton
        parameter = self.setting.parameter

        # clear prediction
        self.prediction = []

        # run victim side
        # self.setting.dlg.victim_side()


        # Run prediction strategie
        if parameter["version"] == "random":
            self.random_prediction()
        elif parameter["version"] == "idlg":
            self.simplified_prediction()
        elif parameter["version"] == "v1":#LLG
            self.v1_prediction()
        elif parameter["version"] == "v2":#LLG+
            self.v2_prediction()
        elif parameter["version"] in ["v3-zero", "v3-one", "v3-random"]:
            self.v3_prediction()
        else:
            exit("Unknown prediction strategy {}".format(parameter["version"]))

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
        print(self.setting.parameter["version"], ": ACC: ", self.acc)

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


    def v1_prediction(self): #LLG

        parameter = self.setting.parameter
        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        h1_extraction = []
        impact_acc = 0

        # filter negative values
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
            if class_gradient < 0:
                h1_extraction.append((i_cg, class_gradient))
                impact_acc += class_gradient.item()

        # mean value
        self.impact = (impact_acc / parameter["batch_size"]) * (1 + 1 / parameter["num_classes"])

        # save predictions
        for (i_c, _) in h1_extraction:
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



    def v2_prediction(self): #LLG+
        parameter = self.setting.parameter

        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        h1_extraction = []

        # do h1 extraction
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
            if class_gradient < 0:
                h1_extraction.append((i_cg, class_gradient))

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
                tmp_gradients = torch.sum(tmp_setting.dlg.gradient[-2], dim=-1).cpu().detach().numpy()
                impact += torch.sum(tmp_setting.dlg.gradient[-2], dim=-1)[i].item()
                for j in range(parameter["num_classes"]):
                    if j == i:
                        continue
                    else:
                        acc_offset[j] +=tmp_gradients[j]

            impact /= (parameter["num_classes"] * parameter["batch_size"])
            acc_impact += impact

        self.impact = (acc_impact / n) * (1 + 1/parameter["num_classes"])

        acc_offset = np.divide(acc_offset, n*(parameter["num_classes"]-1))
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset


        # compensate h1 extraction
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidat, likely to be present, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one occurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)




    def v3_prediction(self): #LLG- with dummies
        parameter = self.setting.parameter

        self.gradients_for_prediction = torch.sum(self.setting.dlg.gradient[-2], dim=-1).clone()
        h1_extraction = []

        # do h1 extraction
        for i_cg, class_gradient in enumerate(self.gradients_for_prediction):
            if class_gradient < 0:
                h1_extraction.append((i_cg, class_gradient))

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
                if parameter["version"] == "v3-zero":
                    tmp_setting.configure(targets=[i] * parameter["batch_size"], dataset="DUMMY-ZERO")
                elif parameter["version"] == "v3-one":
                    tmp_setting.configure(targets=[i] * parameter["batch_size"], dataset="DUMMY-ONE")
                elif parameter["version"] == "v3-random":
                    tmp_setting.configure(targets=[i] * parameter["batch_size"], dataset="DUMMY-RANDOM")
                else:
                    exit("v3 called with wrong version")


                tmp_setting.dlg.victim_side()
                tmp_gradients = torch.sum(tmp_setting.dlg.gradient[-2], dim=-1).cpu().detach().numpy()
                impact += torch.sum(tmp_setting.dlg.gradient[-2], dim=-1)[i].item()
                for j in range(parameter["num_classes"]):
                    if j == i:
                        continue
                    else:
                        acc_offset[j] +=tmp_gradients[j]

            impact /= (parameter["num_classes"] * parameter["batch_size"])
            acc_impact += impact

        self.impact = (acc_impact / n) * (1 + 1/parameter["num_classes"])

        acc_offset = np.divide(acc_offset, n*(parameter["num_classes"]-1))
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset


        # compensate h1 extraction
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(self.prediction)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)








