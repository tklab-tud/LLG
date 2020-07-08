import matplotlib.pyplot as plt


class Graph:
    def __init__(self, setting):
        self.setting = setting
        self.data = []

    def add_datapoint(self, x, y, line):
        self.data.append((x, y, line))

    def show(self):
        self.plot()
        plt.show()

    def plot(self):
        # For every line
        for label in dict.fromkeys([label for (_, _, label) in self.data]):
            l_x = [x for (x, y, l) in self.data if l == label]
            l_y = [y for (x, y, l) in self.data if l == label]
            plt.plot(l_x, l_y, label=label)

        plt.legend()

    def save(self, path_name):
        plt.savefig(path_name)
