import os

import matplotlib.pyplot as plt


class Graph:
    def __init__(self, setting, xlabel, ylabel):
        self.setting = setting
        self.data = []

        self.fig, self.subplot = plt.subplots(1, 1)
        self.subplot.set_xlabel(xlabel)
        self.subplot.set_ylabel(ylabel)

    def add_datapoint(self, x, y, line):
        self.data.append((x, y, line))

    def add_datarow(self, x_l, y_l, label):
        for i in range(len(x_l)):
            self.add_datapoint(x_l[i], y_l[i], label)

    def show(self):
        self.plot()
        self.fig.show()

    def plot(self):
        # For every line
        for label in dict.fromkeys([label for (_, _, label) in self.data]):
            l_x = [x for (x, y, l) in self.data if l == label]
            l_y = [y for (x, y, l) in self.data if l == label]
            self.subplot.plot(l_x, l_y, label=label)

        self.subplot.legend()

    def save(self, path_name):
        self.plot()
        if not os.path.exists(self.setting.parameter["result_path"]):
            os.makedirs(self.setting.parameter["result_path"])
        self.fig.savefig(path_name)
