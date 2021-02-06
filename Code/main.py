from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    new = True
    new = False

    if new:

        dataloader = Dataloader()
        experiment(dataloader=dataloader,
                   list_datasets=["MNIST"],
                   list_bs=[4],
                   list_balanced=[True],
                   list_versions=["v2"],    # v1=LLG, v2=LLG+, "v3-zero", "v3-one", "v3-random"
                   n=100,                     # Amount of attacks
                   extent="victim_side",        # "victim_side", "predict", "reconstruct"
                   trainsize=1,             # Iterations per Trainstep
                   trainsteps=0,           # Number of Attack&Train cycles
                   path=None,
                   reconstruction_steps=0,
                   model="LeNet",
                   store_individual_gradients=True
                   )

    else:

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()


        # Magnitude Check plots a scatterplot of the gradients of a run.
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        # Grads before summing up, after summing up and after adjustment
        magnitude_check(run, path, gradient_type="individual_gradients", group_by_class=True)





    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
