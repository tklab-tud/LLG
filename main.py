from Graph import Graph
from Setting import Setting
from examples import mse_test, accuracy_test
import numpy as np


def main():
    ############## Build your attack here ######################

    setting, graph = mse_test()
    graph.save("results/mse_test.png")

    """
    #Still buggy!
    setting = Setting(dlg_iterations=30,
                      log_interval=3,
                      batch_size=4,
                      use_seed=False,
                      dlg_lr=0.5,
                      )

    graph = Graph(setting, "Iterations", "MSE")

    setting.attack()
    setting.show_composed_image()
    graph.add_mses()
    graph.show()
    """

    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
