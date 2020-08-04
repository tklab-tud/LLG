from examples import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################


    setting, graph = prediction_accuracy_vs_batchsize(1, range(60,129), "CIFAR", False)



    ############################################################
    print("Run finished")

if __name__ == '__main__':
    main()
