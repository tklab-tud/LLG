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
                   list_bs=[2],
                   list_balanced=[True],
                   list_versions=["dlg"],   # v1=LLG, v2=LLG+, "v3-zero", "v3-one", "v3-random", "dlg", "idlg"
                   n=2,                     # Amount of attacks
                   extent="reconstruct",        # "victim_side", "predict", "reconstruct"
                   trainsize=0,             # Iterations per Trainstep
                   trainsteps=0,           # Number of Attack&Train cycles
                   path=None,
                   model="LeNet",
                   store_individual_gradients=True, # Will store the ~500 gradients connected to one output node and not just their sum
                   dlg_lr= 1, # learrate of (i)dlg image reconstruction
                   dlg_iterations= 100, # amount of (i)dlg reconstruction iterations
                   log_interval=10,  # Won't store each (i)dlg iteration's images but every n-th iteration's
                   store_composed_image = True, # storing dlg output images as composed image
                   store_separate_images = True, # storing dlg output images as seperate images

                   )

    else:

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()


        # magnitude_check plots a scatterplot of the gradients of a run.
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        # Grads before summing up, after summing up and after adjustment
        # magnitude_check(run, path, gradient_type="individual_gradients", group_by_class=True)


        # negativ_value_check partitions the gradients into 4 categories: (non)present x sign
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        #
        #negativ_value_check(run, path, gradient_type = "individual_gradients")

        # same_sign_check(run, path, dataset=None, balanced=None)
        # checks the split gradient_sum_sign x individual_grad_sign
        same_sign_check(run, path)



    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
