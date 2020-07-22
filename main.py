from Graph import *
from Setting import *
from examples import *
import numpy as np


def main():
    ############## Build your attack here ######################

    setting, graph = prediction_accuracy_vs_batchsize_line(True)

    graph.show()
    print("done")




    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
