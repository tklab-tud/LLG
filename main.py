from Graph import *
from Setting import *
from examples import *
import numpy as np


def main():
    ############## Build your attack here ######################

    setting, graph = prediction_accuracy_vs_batchsize_line()
    graph.show()
    print("continue")
    setting = Setting.load_json(None)
    graph = Prediction_accuracy_graph(setting[0], "Batch-Size", "Accuracy")
    for s in setting:
        graph.setting = s
        graph.add_prediction_acc(s.parameter["prediction"], s.parameter["batch_size"])

    graph.plot_line()
    graph.show()
    print("done")



    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
