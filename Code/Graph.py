import os

import matplotlib.pyplot as plt
import numpy as np


class Graph:
    def __init__(self, setting, xlabel, ylabel):
        self.setting = setting
        self.data = []

        self.fig, self.subplot = plt.subplots(1, 1)
        self.subplot.set_xlabel(xlabel)
        self.subplot.set_ylabel(ylabel)
        self.subplot.use_sticky_edges = True

    def add_datapoint(self, line, y, x=0):
        self.data.append((line, y, x,))

    def add_datarow(self, label, y_l, x_l=None):
        if x_l is None: x_l = [0] * len(y_l)
        for i in range(len(x_l)):
            self.add_datapoint(label, y_l[i], x_l[i])

    def show(self):
        self.fig.show()

    def plot_bar(self):
        self.take_average()
        for dat in self.data:
            self.subplot.bar(dat[0], dat[1], 0.5, color="blue")

    def plot_line(self, style='solid', clear=True, color=None):
        if clear: self.subplot.clear()
        self.take_average()
        # For every line
        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            self.subplot.plot(l_x, l_y, label=label, linestyle=style, color=color)

        self.subplot.legend()

    def take_average(self):
        # Repaces data with the averages for every label,y combination
        labels = [str(x[0]) for x in self.data]
        ys = [x[1] for x in self.data]
        xs = [x[2] for x in self.data]

        # prepare dict
        dict = {}
        for label, y, x in zip(labels, ys, xs):
            if dict.keys().__contains__(label):
                if dict[label].keys().__contains__(x):
                    dict[label][x]["value"] += y
                    dict[label][x]["count"] += 1
                else:
                    dict[label].update({x: {"value": y, "count": 1}})
            else:
                dict.update({label: {x: {"value": y, "count": 1}}})

        # everything is in dict, reset data and fill again with averaged data

        self.data = []

        for label in dict.keys():
            for x in dict[label]:
                self.data.append((label, dict[label][x]["value"] / dict[label][x]["count"], x))

    def save(self, path_name):
        if not os.path.exists(self.setting.parameter["result_path"]):
            os.makedirs(self.setting.parameter["result_path"])
        self.fig.savefig(self.setting.parameter["result_path"] + "/" + path_name)


class Mses_vs_Iterations_graph(Graph):
    def __init__(self, setting, xlabel, ylabel):
        super().__init__(setting, xlabel, ylabel)

    def add_all_mses(self, label):
        mses_unprocessed = self.setting.result.mses
        mses = []
        for step in mses_unprocessed:
            mses.append(np.mean(step))

        li = self.setting.parameter["log_interval"]
        self.add_datarow(label, mses, list(range(li, len(mses) * li + 1, li)))


    def add_last_mse(self, label, x=0):
        mses_unprocessed = self.setting.result.mses
        mses = []
        for step in mses_unprocessed:
            mses.append(np.mean(step))

        li = self.setting.parameter["log_interval"]
        self.add_datarow(label, mses[-1], x)


class Prediction_accuracy_graph(Graph):
    def __init__(self, setting, xlabel, ylabel):
        super().__init__(setting, xlabel, ylabel)

    def add_prediction_acc(self, label, x):
        acc = self.setting.predictor.acc
        self.add_datapoint(label, acc, x)
