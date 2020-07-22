from Graph import *
from Setting import *
from examples import *
import numpy as np


def main():
    ############## Build your attack here ######################

    setting, graph = mse_vs_batchsize_line()

    graph.show()
    print("done")




    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
