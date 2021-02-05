from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    new = True
    #new = False

    if new:

        dataloader = Dataloader()
        experiment(dataloader=dataloader,
                   list_datasets=["MNIST"],
                   list_bs=[4],
                   list_balanced=[True],
                   list_versions=["v2"],    # v1=LLG, v2=LLG+, "v3-zero", "v3-one", "v3-random"
                   n=1,                     # Amount of attacks
                   extent="predict",        # "victim_side", "predict", "reconstruct"
                   trainsize=1,             # Iterations per Trainstep
                   trainsteps=10,           # Number of Attack&Train cycles
                   path=None,
                   reconstruction_steps=0,
                   model="LeNet"
                   )

    else:

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()

        for adjusted in [False]:
            magnitude_check(run, path, adjusted=adjusted, group_by_class=True)

        ### Some examples below ###

        #visualize_class_prediction_accuracy_vs_batchsize(run, path)

        #negativ_value_check(run, path, dataset="MNIST", balanced=True)
        #negativ_value_check(run, path, dataset="MNIST", balanced=False)
        #negativ_value_check(run, path, dataset="CIFAR", balanced=True)
        #negativ_value_check(run, path, dataset="CIFAR", balanced=False)
        #pearson_check(run, path, version="v2")

        """
        for dataset in ["MNIST", "CIFAR", "CELEB-A-male"]:
            for adjusted in [True, False]:
                magnitude_check(run, path, adjusted=adjusted, balanced=True, dataset=dataset)
        """

        """
        for trainstep in range(0, 10, 1):
            for adjusted in [False]:
                for version in ["v2"]:
                    magnitude_check(run, path, adjusted=adjusted, version=version, group_by_class=True, trainstep=trainstep)
        """

        #heatmap(run, path, adjusted=False, balanced=True, version="v2", dataset="MNIST", list_bs=[32])

        #visualize_hellinger_vs_batchsize(run, path)


        #"""
        #visualize_class_prediction_accuracy_vs_batchsize(run, path)

        #visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=None, dataset=None, version=None):



            #visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path)
        #"""

        visualize_class_prediction_accuracy_vs_training(run, path)

        #visualize_good_fidelity(run, path, [0.1, 0.05, 0.01, 0.005, 0.001], 4, True)




    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
