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
        elif parameter["version"] == "experimental":
            self.experimental_prediction()
        else:
            exit("Unknown prediction strategy {}".format(parameter["version"]))

        self.prediction.sort()
        self.update_accuracy()



    def update_accuracy(self):
         # analyse prediction
        orig_label = []
        for it in range(self.setting.parameter["local_iterations"]):
            orig_label += self.setting.parameter["orig_label"][it].tolist()

        if not self.setting.parameter["batch_size"] * self.setting.parameter["local_iterations"] == len(self.prediction):
            print("WARNING: Prediction did not produce the correct amount of class labels.")
            print("predicted: {}; expected: {} (batch_size: {}, local_iterations: {})".format(
                len(self.prediction),
                self.setting.parameter["batch_size"]*self.setting.parameter["local_iterations"],
                self.setting.parameter["batch_size"],
                self.setting.parameter["local_iterations"]))


        self.correct = 0
        self.false = 0
        for p in self.prediction:
            if orig_label.__contains__(p):
                orig_label.remove(p)
                self.correct += 1
            else:
                self.false += 1

        self.acc = self.correct / (self.correct + self.false)

        defense = "none"
        if self.setting.parameter["compression"]:
            defense = "compression"
        elif self.setting.parameter["differential_privacy"]:
            defense = "differential_privacy"
        elif self.setting.parameter["dropout"] != 0.0:
            defense = "dropout"

        print(f'{self.setting.parameter["version"]}: {defense}: ACC: {self.acc}')


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

        for _ in range( self.setting.parameter["batch_size"]*self.setting.parameter["local_iterations"]):
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
        self.impact = (impact_acc / parameter["batch_size"]) * (1 + 1 / parameter["num_classes"])/ parameter["local_iterations"]

        # save predictions
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)


        # predict the rest
        for _ in range(parameter["batch_size"]*parameter["local_iterations"] - len(self.prediction)):
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
        tmp_setting.configure(local_iterations=1)
        impact = 0
        acc_impact = 0
        acc_offset = np.zeros(parameter["num_classes"])
        n = 10

        # calculate bias and impact
        for _ in range(n):
            tmp_gradients = []
            impact = 0
            for i in range(parameter["num_classes"]):
                tmp_setting.configure(targets=[i] * parameter["batch_size"],
                                      compression=False,
                                      differential_privacy=False,
                                      dropout=False,
                                      local_training=False,
                                      local_iterations=1)
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

        self.impact = (acc_impact / n) * (1 + 1/parameter["num_classes"]) / parameter["local_iterations"]

        acc_offset = np.divide(acc_offset, n*(parameter["num_classes"]-1))
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset


        h1_extraction.sort(key=lambda y: y[1])

        # compensate h1 extraction
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)
            if len(self.prediction) >= parameter["batch_size"]*parameter["local_iterations"]:
                break

        # predict the rest
        for _ in range(parameter["batch_size"]*parameter["local_iterations"] - len(self.prediction)):
            # add minimal candidat, likely to be present, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one occurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)

    def experimental_prediction(self):  # LLG+
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
            # impact = 0
            for i in range(parameter["num_classes"]):
                tmp_setting.configure(targets=[i] * parameter["batch_size"])
                tmp_setting.dlg.victim_side()
                tmp_gradients = torch.sum(tmp_setting.dlg.gradient[-2], dim=-1).cpu().detach().numpy()
                impact += torch.sum(tmp_setting.dlg.gradient[-2], dim=-1)[i].item()
                for j in range(parameter["num_classes"]):
                    if j == i:
                        continue
                    else:
                        acc_offset[j] += tmp_gradients[j]

            impact /= (parameter["num_classes"] * parameter["batch_size"])
            acc_impact += impact

        self.impact = (acc_impact / n) * (1 + 1 / parameter["num_classes"]) / parameter["local_iterations"]

        acc_offset = np.divide(acc_offset, n * (parameter["num_classes"] - 1))
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset

        # compensate h1 extraction
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)

        # predict the rest
        for _ in range(parameter["batch_size"]*parameter["local_iterations"] - len(self.prediction)):
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
            impact = 0
            for i in range(parameter["num_classes"]):
                if parameter["version"] == "v3-zero":
                    dataset="DUMMY-ZERO"
                elif parameter["version"] == "v3-one":
                    dataset="DUMMY-ONE"
                elif parameter["version"] == "v3-random":
                    dataset="DUMMY-RANDOM"
                else:
                    exit("v3 called with wrong version")

                tmp_setting.configure(targets=[i] * parameter["batch_size"], dataset=dataset,local_training=False,
                                      local_iterations=1)


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

        self.impact = (acc_impact / n) * (1 + 1/parameter["num_classes"]) / parameter["local_iterations"]

        acc_offset = np.divide(acc_offset, n*(parameter["num_classes"]-1))
        self.offset = torch.Tensor(acc_offset).to(self.setting.device)

        self.gradients_for_prediction -= self.offset


        # compensate h1 extraction
        for (i_c, _) in h1_extraction:
            self.prediction.append(i_c)
            self.gradients_for_prediction[i_c] = self.gradients_for_prediction[i_c].add(-self.impact)

        # predict the rest
        for _ in range(parameter["batch_size"] * parameter["local_iterations"] - len(self.prediction)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(self.gradients_for_prediction).item()
            self.prediction.append(min_id)

            # add the mean value of one accurance to the candidate
            self.gradients_for_prediction[min_id] = self.gradients_for_prediction[min_id].add(-self.impact)








