from Graph import Graph
from Setting import Setting
from examples import mse_test, accuracy_test


def main():
    ############## Build your attack here ######################

    setting, graph = mse_test()
    graph.save(setting.parameter["result_path"]+"mse_test.jpg")
    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
