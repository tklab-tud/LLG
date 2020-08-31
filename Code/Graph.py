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

    def plot_bar(self, alt_ax=False):
        if alt_ax:
            plt = self.subplot2
        else:
            plt = self.subplot

        self.take_average()
        for dat in self.data:
            plt.bar(str(dat[0]), dat[1], 0.5, color=self.color(str(dat[0])))

    def plot_line(self, alt_ax=False):
        if alt_ax:
            plt = self.subplot2
        else:
            plt = self.subplot

        self.take_average()
        # For every line
        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x.lstrip("0") for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            color, style, _ = self.color(label)
            plt.plot(l_x, l_y, label=label, linestyle=style, color=color)

        handles, labels = plt.get_legend_handles_labels()

        order = []
        for label in labels:
            order.append(self.color(label)[2])
        labels, handles, order = zip(*sorted(zip(labels, handles, order), key=lambda t: t[2]))

        plt.legend(handles, labels, prop={'size': 11})

    def plot_scatter(self):
        plt = self.subplot
        max_x = 0

        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            max_x = max(max_x, max(l_x))
            color, style, _ = self.color(str(label))
            plt.scatter(l_x, l_y, label="Batch Size: " + str(label), marker=style, edgecolors=color, facecolors="none")

        plt.set_xticks(range(0, max_x + 1, max(1, max_x // 10)))

        plt.legend(loc="lower right")

    def plot_heatmap(self):



        x_max = max([x for _, _, x in self.data])
        x_min = min([x for _, _, x in self.data])
        y_max = max([y for _, y, _ in self.data])
        y_min = min([y for _, y, _ in self.data])
        y_span = y_max - y_min
        x_span = x_max - x_min

        heat = np.zeros((x_max+1, 101 ))

        for _, y, x in self.data:
            heat_y = 100 - (int((y - y_min) // (y_span / 100)))
            heat[x][heat_y] = min(heat[x][heat_y]+1, 1000)

        heat = np.transpose(heat)

        plt.imshow(heat, cmap='hot', interpolation='bilinear', extent=[0,x_max,y_min,y_max], aspect=0.05)

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

    def sort(self):
        self.data.sort()

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(path + name)

    def save_f(self, f):
        if f is None:
            return
        self.fig.savefig(f.name)

    def color(self, s):
        color = {
            "1": "#e6194B", "2": '#800000', "4": '#f58231', "8": '#3cb44b', "16": '#4363d8',
            "32": '#000075', "64": '#911eb4', "128": '#000000', "256": '#f032e6',

            "Random (IID)": "#e6194B",
            "LLG (IID)": '#3cb44b',
            "LLG+ (IID)": '#4363d8',
            "Random (non-IID)": '#f58231',
            "LLG (non-IID)": '#42d4f4',
            "LLG+ (non-IID)": '#f032e6',

        }
        symbol = {
            "1": ".", "2": '*', "4": '8', "8": 'v', "16": 's',
            "32": 'p', "64": 'D', "128": 'P', "256": '^',

            "Random (IID)": "--",
            "LLG (IID)": (0, (3, 5, 1, 5, 1, 5)),
            "LLG+ (IID)": '-',
            "Random (non-IID)": ':',
            "LLG (non-IID)": (0, (3, 1, 1, 1)),
            "LLG+ (non-IID)": '-.',

        }
        order = {
            "1": 1, "2": 2, "4": 3, "8": 4, "16": 5,
            "32": 6, "64": 7, "128": 8, "256": 9,

            "Random (IID)": 4,
            "LLG (IID)": 2,
            "LLG+ (IID)": 0,
            "Random (non-IID)": 5,
            "LLG (non-IID)": 3,
            "LLG+ (non-IID)": 1,

        }

        return color[s], symbol[s], order[s]
