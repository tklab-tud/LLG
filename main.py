import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np



ap = argparse.ArgumentParser(description="Comparison framework for attacks on federated learning.")

ap.add_argument("-d", "--dataset", required=False, default="MNIST", help="Dataset to use", choices=["MNIST", "SVHN", "CIFAR", "ATT"])
ap.add_argument("-a", "--attack", required=False, default="GAN", help="Attacks to perform", choices=["GAN", "MI", "UL", "DLG", "iDLG"])
ap.add_argument("-n", "--nodes", required=False, default=10, type=int , help="Amount of noodes")
ap.add_argument("-s", "--size", required=False, default=10, type=int, help="Amount of samples per node")
ap.add_argument("-r", "--rounds", required=False, default=10 , help="Amount of learning rounds")
ap.add_argument("-c", "--cpr", required=False, default=10 , help="Amount of clients per learning round")
ap.add_argument("-S", "--selective", required=False, default=False , help="Selective up- and download", action='store_true')

args = vars(ap.parse_args())

for arg in args:
    print(str(arg)+" "+str(args[arg]))

###############################################
