import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import itertools


class Graph:
    def __init__(self, xlabel, ylabel, ylabel2=None, y_range=None, fontsize=14):
        self.data = []
        self.fig, self.subplot = plt.subplots(1, 1) # figsize=(8,5)
        self.fontsize = fontsize
        self.subplot.set_xlabel(xlabel, fontsize=self.fontsize)
        self.subplot.set_ylabel(ylabel, fontsize=self.fontsize)
        if ylabel2 is not None:
            self.subplot2 = self.subplot.twinx()
            self.subplot2.set_ylabel(ylabel2)
            self.fig.tight_layout()
        else:
            self.subplot2 = None

        self.subplot.use_sticky_edges = True
        self.y_range = y_range

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

    def plot_line(self, alt_ax=False, location="lower right", move=None, legend=True, skip_x_ticks=False):
        if alt_ax:
            plt = self.subplot2
        else:
            plt = self.subplot

        self.take_average()
        # For every line
        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x.lstrip("0") for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            color = self.color(label)
            style = self.style(label)

            print("Min y: {} for label {}".format(str(min(l_y)), label))
            plt.plot(l_x, l_y, label=label, linestyle=style, color=color, linewidth=2.5)

        # plt.ticklabel_format(scilimits=(0,3),useMathText=True)
        # plt.set_ylim(self.y_range)
        # plt.set_xlim([-5,105])
        plt.tick_params(axis='x', labelsize=self.fontsize)
        plt.tick_params(axis='y', labelsize=self.fontsize)

        handles, labels = plt.get_legend_handles_labels()

        if legend:
            order = []
            for label in labels:
                order.append(self.order(label))
            labels, handles, order = zip(*sorted(zip(labels, handles, order), key=lambda t: t[2]))
            plt.legend(handles, labels, prop={'size': self.fontsize}, loc=location, bbox_to_anchor=move)
        if skip_x_ticks:
            plt.set_xticks(range(0, len(self.data), max(1, len(self.data) // 10)), fontsize=self.fontsize)

    def plot_scatter(self, location="best", move=None, legend=True):
        if self.data == []:
            return

        plt = self.subplot
        max_x = 0

        for label in dict.fromkeys([label for (label, _, _) in self.data]):
            l_x = [x for (l, y, x) in self.data if l == label]
            l_y = [y for (l, y, x) in self.data if l == label]
            max_x = max(max_x, max(l_x))

            color = self.color((str(label)))


            style = self.style(str(label))
            if len(l_y) == 1:
                print("Cant calculate pearson from n=1 values")
                pearson_r = [0]
            else:
                pearson_r = scipy.stats.pearsonr(l_x, l_y)

            visible_label = str(label) + ", $\\rho = {:.5f}$".format(pearson_r[0])
            plt.scatter(l_x, l_y, label=visible_label, marker=style, edgecolors=color, facecolors="none")

        plt.set_xticks(range(0, max_x + 1, max(1, max_x // 10)), fontsize=self.fontsize)
        plt.margins(x=0)

        if legend:
            order = []
            handles, labels = plt.get_legend_handles_labels()
            for label in labels:
                order.append(self.order(label.split(",")[0]))
            labels, handles, order = zip(*sorted(zip(labels, handles, order), key=lambda t: t[2]))
            plt.legend(handles, labels, prop={'size': self.fontsize}, loc=location, bbox_to_anchor=move)
            if self.y_range is not None:
                plt.set_ylim(self.y_range[0], self.y_range[-1])

    def plot_heatmap(self):

        x_max = max([x for _, _, x in self.data])
        x_min = min([x for _, _, x in self.data])
        if self.y_range is None:
            self.y_range = [0,0]
            self.y_range[1] = max([y for _, y, _ in self.data])
            self.y_range[0] = min([y for _, y, _ in self.data])
        y_max = self.y_range[1]
        y_min = self.y_range[0]
        y_span = y_max - y_min
        x_span = x_max - x_min

        heat = np.zeros((x_max + 1, 101))

        for _, y, x in self.data:
            if not self.y_range[0] < y < self.y_range[1]: # a value outside of y_range
                print("Skipping {}, because its outside y_range [{}, {}]".format(y, self.y_range[0], self.y_range[1]))
                continue
            heat_y = 100 - (int((y - y_min) // (y_span / 100)))
            heat[x][heat_y] = min(heat[x][heat_y] + 1, 1000)

        heat = np.transpose(heat)
        plt.xticks(range(0, x_max + 1, max(1, x_max // 10)), fontsize=self.fontsize)
        if self.y_range is not None:
            plt.ylim(self.y_range[0], self.y_range[-1])

        plt.imshow(heat, cmap='hot', interpolation='spline16', extent=[0, x_max, y_min, y_max], aspect='auto')

        norm = matplotlib.colors.Normalize(vmin=np.min(heat), vmax=np.max(heat), clip=False)
        cbar = self.subplot.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hot'))
        cbar.ax.set_ylabel("Number of gradients", rotation=-90, va="bottom")



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

        # sorted(dict.items(), key = lambda kv: kv[1])

        for label in dict.keys():
            # for x in sorted(dict[label]):
            for x in dict[label]:
                self.data.append((label, dict[label][x]["value"] / dict[label][x]["count"], x))

    def sort(self):
        self.data.sort()

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(path + name, dpi=600, format='pdf')

    def save_f(self, f):
        if f is None:
            return
        self.fig.savefig(f.name, dpi=600, format='pdf')

    def color(self, s):
        color = {
            "bs1": "#e6194B", "bs2": '#800000', "bs4": '#f58231', "bs8": '#3cb44b', "bs16": '#4363d8',
            "bs32": '#000075', "bs64": '#911eb4', "bs128": '#000000', "bs256": '#f032e6',

            "class0": "#e6194B", "class1": '#800000', "class2": '#f58231', "class3": '#3cb44b', "class4": '#4363d8',
            "class5": '#000075', "class6": '#911eb4', "class7": '#000000', "class8": '#f032e6', "class9": "#ff0044",

            "none": "#e6194B", "compression": '#800000', "dp": '#f58231', "dropout": '#3cb44b',

            "Random": "#e6194B",
            "LLG": "#911eb4",
            "LLG+": '#4363d8',
            "LLG*": '#f032e6',
            "iDLG": "#800000",
            "DLG": '#277831', #'#3cb44b',

            "CNN": '#4363d8',
            "LeNet": '#277831',
            "OldLeNet": '#ff0044',
            "FCNN": '#f032e6',
            "ResNet": "#e6194B",

            # compression threshold
            "θ=0%": "#4363d8",
            "θ=10%": "#ff0044",
            "θ=20%": '#277831',
            "θ=40%": '#f032e6',
            "θ=80%": '#f58231',

            # noise multiplier
            "var = 10⁻⁴": "#e6194B",
            "var = 10⁻³": '#f032e6',
            "var = 10⁻²": '#4363d8',
            "var = 10⁻¹": '#277831',
            "var = 0": "#4363d8",
            "basline": "#4363d8",
            "1.0": "#e6194B",
            "0.5": '#f032e6',
            "0.25": '#4363d8',
            "0.75": '#f032e6',
            "1.5": "#e6194B",
            "2.0": '#4363d8',

            "norm = inf": "#4363d8",
            "norm = 0": "#4363d8",
            "norm = 10⁻⁴": '#277831',
            "norm = 10⁻³": '#4363d8',
            "norm = 10⁻²": '#f032e6',
            "norm = 10⁻¹": "#e6194B",

            # noise type
            "normal": "#e6194B",
            "laplace": '#277831',
            "exponential": '#4363d8',
            "none": '#f032e6',

            "Random (IID)": "#e6194B",
            "LLG (IID)": '#277831',
            "LLG+ (IID)": '#4363d8',
            "iDLG (IID)": "#800000",
            "DLG (IID)": "#911eb4",
            "Random (non-IID)": '#f58231',
            "LLG (non-IID)": '#42d4f4',
            "LLG+ (non-IID)": '#f032e6',
            "iDLG (non-IID)": "#000075",
            "DLG (non-IID)": "#808000",
            "LLG-ZERO (IID)": "#b09ae6",
            "LLG-ZERO (non-IID)": "#f57de3",
            "LLG-ONE (IID)": "#8f5e10",
            "LLG-ONE (non-IID)": "#9c0676",
            "LLG-RANDOM (IID)": "#ff0044",
            "LLG-RANDOM (non-IID)": "#ffbf00",


            "Model": '#000000' #"#FF0F0F"

        }

        return color[s] if color.__contains__(s) else "#000000"

    def style(self, s):
        style = {
            "bs1": ".", "bs2": '*', "bs4": '8', "bs8": 'v', "bs16": 's',
            "bs32": 'p', "bs64": 'D', "bs128": 'P', "bs256": '^',

            "class0": ".", "class1": '*', "class2": '8', "class3": 'v', "class4": 's',
            "class5": 'p', "class6": 'D', "class7": 'P', "class8": '^', "class9": 'h',

            "none": ".", "compression": '*', "dp": '8', "dropout": 'v',

            "Random": "--",
            "LLG": (0, (3, 1, 1, 1, 1, 1)),
            "LLG+": '-',
            "LLG*": '-.',
            "iDLG": (0, (3, 10, 1, 10, 1, 10)),
            "DLG": (0, (3, 5, 1, 5, 1, 5)),

            "CNN": '-',
            "LeNet": (0, (3, 5, 1, 5, 1, 5)),
            "OldLeNet": (0, (1, 1, 1, 3)),
            "FCNN": '-.',
            "ResNet": "--",

            # compression threshold
            "θ=0%": '-',
            "θ=10%": (0, (1, 1, 1, 3)),
            "θ=20%": (0, (3, 5, 1, 5, 1, 5)),
            "θ=40%": '-.',
            "θ=80%": ':',

            # noise multiplier
            "var = 10⁻⁴": ':',
            "var = 10⁻³": '-.',
            "var = 10⁻²": (0, (3, 5, 1, 5, 1, 5)),
            "var = 10⁻¹": (0, (1, 1, 1, 3)),
            "var = 0": '-',
            "basline": '-',
            "1.0": ':',
            "0.5": '-.',
            "0.25": (0, (3, 5, 1, 5, 1, 5)),
            "0.75": (0, (3, 5, 1, 5, 1, 5)),
            "1.5": '-.',
            "2.0": ':',

            "norm = inf": '-',
            "norm = 0": '-',
            "norm = 10⁻⁴": ':',
            "norm = 10⁻³": '-.',
            "norm = 10⁻²": (0, (3, 5, 1, 5, 1, 5)),
            "norm = 10⁻¹": (0, (1, 1, 1, 3)),

            # noise type
            "normal": "--",
            "laplace": (0, (3, 5, 1, 5, 1, 5)),
            "exponential": '-',
            "none": '-.',

            "Random (IID)": "--",
            "LLG (IID)": (0, (3, 5, 1, 5, 1, 5)),
            "LLG+ (IID)": '-',
            "iDLG (IID)": (0, (3, 10, 1, 10, 1, 10)),
            "DLG (IID)": (0, (3, 1, 1, 1, 1, 1)),
            "Random (non-IID)": ':',
            "LLG (non-IID)": (0, (3, 1, 1, 1)),
            "LLG+ (non-IID)": '-.',
            "iDLG (non-IID)": (0, (1, 4, 10, 1)),
            "DLG (non-IID)": (0, (1, 1, 1, 3)),

            "Model": "solid"

        }
        return style[s] if style.__contains__(s) else "-"

    def order(self, s):
        order = {
            "bs1": 1, "bs2": 2, "bs4": 3, "bs8": 4, "bs16": 5,
            "bs32": 6, "bs64": 7, "bs128": 8, "bs256": 9,

            "LLG": 0,
            "LLG*": 2,
            "LLG+": 4,
            "DLG": 6,
            "iDLG": 8,
            "Random": 10,

            "CNN": 0,
            "FCNN": 2,
            "LeNet": 4,
            "OldLeNet": 6,
            "ResNet": 8,

            # compression threshold
            "θ=0%": 0,
            "θ=10%": 1,
            "θ=20%": 2,
            "θ=40%": 4,
            "θ=80%": 8,

            # noise multiplier
            "var = 0": 0,
            "var = 10⁻⁴": 1,
            "var = 10⁻³": 2,
            "var = 10⁻²": 3,
            "var = 10⁻¹": 4,
            "0.25": 5,
            "0.5": 6,
            "0.75": 7,
            "1.0": 8,
            "1.5": 9,
            "2.0": 10,

            "norm = inf": 0,
            "norm = 0": 0,
            "norm = 10⁻⁴": 1,
            "norm = 10⁻³": 2,
            "norm = 10⁻²": 3,
            "norm = 10⁻¹": 4,

            # noise type
            "normal": 1,
            "laplace": 2,
            "exponential": 3,
            "none": 100,

            "Random (IID)": 4,
            "LLG (IID)": 2,
            "LLG+ (IID)": 0,
            "iDLG (IID)": 8,
            "DLG (IID)": 6,

            "Random (non-IID)": 5,
            "LLG (non-IID)": 3,
            "LLG+ (non-IID)": 1,
            "iDLG (non-IID)": 9,
            "DLG (non-IID)": 7,

            "Model": 100

        }

        return order[s] if order.__contains__(s) else np.random.randint(101, 100000)
