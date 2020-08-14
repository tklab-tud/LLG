import os

import matplotlib.pyplot as plt
import numpy as np


class Graph:
    def __init__(self, xlabel, ylabel, ylabel2=None):
        self.data = []
        self.fig, self.subplot = plt.subplots(1, 1)
        self.subplot.set_xlabel(xlabel)
        self.subplot.set_ylabel(ylabel)
        if ylabel2 is not None:
            self.subplot2 = self.subplot.twinx()
            self.subplot2.set_ylabel(ylabel2)
            self.fig.tight_layout()
        else:
            self.subplot2 = None

        self.subplot.use_sticky_edges = True

    def add_datapoint(self, line, y, x=0):
        self.data.append((line, y, x,))

    def add_datarow(self, label, y_l, x_l=None):
        if x_l is None: x_l = [0] * len(y_l)
        for i in range(len(x_l)):
            self.add_datapoint(label, y_l[i], x_l[i])

    def show(self):
        self.fig.show()

    def plot_bar(self, color=None, alt_ax=False):
        if alt_ax:
            plt = self.subplot2
        else:
            plt = self.subplot

        self.take_average()
        for dat in self.data:
            plt.bar(str(dat[0]), dat[1], 0.5, color=color)

    def plot_line(self, style='solid', color=None, alt_ax=False):
        if alt_ax:
            plt = self.subplot2
        else:
            plt = self.subplot

        self.take_average()
        # For every line
        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            plt.plot(l_x, l_y, label=label, linestyle=style, color=color)

        plt.legend()

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

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(path + name)

    def save_f(self, f):
        if f is None:
            return
        self.fig.savefig(f.name)
