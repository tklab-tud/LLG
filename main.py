from Graph import *
from Setting import *
from examples import *
import numpy as np


def main():
    ############## Build your attack here ######################

    setting, graph = prediction_accuracy_vs_batchsize_line(False)
    """

    setting = Setting.load_json(None)
    graph = Mses_vs_Iterations_graph(setting[0], "Iterations", "MSE")
    for s in setting:
        graph.setting = s
        if s.parameter["improved"]:
            label = s.parameter["prediction"]
        else:
            label = "dlg"

        graph.add_all_mses(label)
    graph.plot_line()
    """

    graph.show()
    print("done")




    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
