from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader
import argparse
import time


def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for LLG Experiment")

    # required arguments
    parser.add_argument('-s', '--set', type=int, default=2,
                            help='experiment set')
    parser.add_argument('-p', '--plot', type=int, default=None,
                            help='number of files to be ploted')
    parser.add_argument('-j', '--job', type=str, default='experiment',
                            help='job to execute. either "experiment" or "visualize".')
    parser.add_argument('-d', '--dir', type=str, default=None,
                            help='directory to plot from.')

    # optional arguments
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                            help='cuda_id to use, if available')

    args = parser.parse_args()

    if args.plot != None:
        args.job = "visualize"
    if args.dir != None:
        args.job = "visualize"

    return args


def main():
    # load parameters
    args = args_parser()
    job = args.job

    # visualization parameters
    # specify the number of json files you want to select for plotting
    num_files = args.plot

    # experiment parameters
    experiment_set = args.set
    cuda_id = args.gpu_id

    # defaults to fill vars, when unused
    extent="predict"
    train_lr = 0.1
    defenses = ["none"]
    differential_privacy = False
    noise_type = "normal"
    sigmas = [0.0]
    betas = [None]
    compression = False
    thetas = [0.0]
    n = 100
    local_training = False
    federated = False
    num_users = 1

    # with exception of set 1 and 3 all sets use non-IID
    balanced = False

    if experiment_set in [1, 3]:
        balanced = True

    # with exception of set 1, 2, 3, and 4 all sets use MNIST and LLG+
    dataset = "MNIST"
    version = "LLG+"

    # with exception of set 3 and 4 all sets use all batch sizes and don't train
    list_bs = [1, 2, 4, 8, 16, 32, 64, 128]
    trainsize = 0
    trainsteps = 0
    local_iterations = 1
    id = "set_"+str(experiment_set)

    LLGs_variant = {"MNIST": "LLG*-zero", "CIFAR": "LLG*-one", "CELEB-A-male": "LLG*-zero", "CELEB-A-hair": "LLG*-zero",
                    "SVHN": "LLG*-random"}

    # FedAvg
    fedAvg = True
    if fedAvg:
        local_training = True
        local_iterations = 10
    else:
        local_training = False
        local_iterations = 1

    if experiment_set == 0:
        local_iterations = 4
        n = 1
        list_bs = [1,4,8]
        dataset = "MNIST"
        balanced = True
        # version = ["LLG", "LLG+", LLGs_variant[dataset], "DLG", "Random"]
        version = ["LLG+", "DLG"]
        model = ["ResNet"]
        local_training = False

    # Set 1 and 2 batch sizes
    if experiment_set in [1, 2, 3, 4]:
        dataset = "MNIST"
        # dataset = "EMNIST"
        # dataset = "FEMNIST"
        # dataset = "FEMNIST-digits"
        # dataset = "CIFAR"
        # dataset = "CELEB-A-male"
        # dataset = "CELEB-A-hair"
        # dataset = "SVHN"
        # versions in order of importance
        version = "LLG+"
        # version = "Random"
        # version = "LLG"
        # version = LLGs_variant[dataset]
        # version = "DLG"
        idx = dataset+"_"+version

    # Set 3 and 4 trained model
    if experiment_set in [3, 4]:
        list_bs = [8]
        trainsize = 100
        trainsteps = 100

    # with exception of set 5 all sets use the ConvNet model
    model = "CNN"

    # Set 5 model architecture comparison
    if experiment_set == 5:
        model = "CNN"
        # model = "LeNet"
        # model = "ResNet"
        # model = "MLP"

        version = "DLG"
        # version ="LLG+"
        idx = model+"_"+version

    # Set 6 additive noise (untrained)
    if experiment_set == 6:
        defenses = ["dp"]
        differential_privacy = True
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"

        sigmas = [0.01, 0.1, 1]
        idx = sigmas[0]

    # Set 7 compression (untrained)
    if experiment_set == 7:
        defenses = ["compression"]
        compression = True
        thetas = [0.2, 0.4, 0.8]
        idx = thetas[0]

    # Set 8 differential privacy (untrained)
    if experiment_set == 8:
        defenses = ["dp"]
        differential_privacy = True
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"

        sigmas = [0.1]
        # betas = [1, 5, 10]
        betas = [10]
        idx = betas[0]

    # Set 9 federated training and trained defenses
    if experiment_set == 9:
        n=1
        extent="victim_side"
        list_bs = [8]
        trainsize = int(10/local_iterations)
        trainsteps = 100
        federated = True
        num_users = 100
        id = "base"
        idx = "FedAvg" if local_training else "FedSGD"
        # TEMP: defense experiments with training
        defenses = ["dp"]
        differential_privacy = True
        # noise_type = "normal"
        # sigmas = [0.01, 0.1, 1]
        # sigmas = [0.01]
        # id = "noise"
        # idx = sigmas[0]
        # betas = [1, 5, 10]
        betas = [10]
        sigmas = [0.1]
        id = "dp"
        idx = betas[0]
        # defenses = ["compression"]
        # compression = True
        # # thetas = [0.2, 0.4, 0.8]
        # thetas = [0.2]
        # id = "comp"
        # idx = thetas[0]
        version = "LLG+"
        # version = "Random"
        # version = "LLG"
        # version = LLGs_variant[dataset]
        # version = "DLG"
        dataset = "MNIST"
        # dataset = "CIFAR"
        # dataset = "CELEB-A-hair"
        # dataset = "SVHN"
        model = "CNN"
        # model = "LeNet"
        # model = "ResNet"
        # model = "MLP"

    if experiment_set == -1:
        n = 1000
        versions = ["LLG+", "Random"]
    elif isinstance(version, list):
        versions = version
    else:
        versions = [version]

    # Set 10 local iterations
    if experiment_set == 10:
        local_iterations_list = [1, 10, 100, 1000]

    start = time.time()

    if job == "experiment":
        dataloader = Dataloader()
        for sigma in sigmas:
            for beta in betas:
                if beta != None and beta > 0:
                    noise_multiplier = sigma / beta
                for theta in thetas:
                # for local_iterations in local_iterations_list:
                    # theta = 0.0
                    experiment(dataloader=dataloader,
                               list_datasets=dataset if isinstance(dataset, list) else [dataset],
                               list_bs=list_bs,
                               list_balanced=[balanced],
                               list_versions=versions,
                               # "LLG"(LLG), "LLG+"(LLG+), "LLG*-zero", "LLG*-one", "LLG*-random", "DLG", "iDLG"
                               n=n,  # Amount of attacks
                               extent=extent,  # "victim_side", "predict", "reconstruct"
                               trainsize=trainsize,  # Iterations per Trainstep
                               trainsteps=trainsteps,  # Number of Attack&Train cycles
                               train_lr=train_lr,
                               federated=federated,
                               num_users=num_users,
                               path="results/{}/{}/{}/{}/".format("FedAvg" if local_training else "FedSGD", id, idx, str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
                               model=model,
                               store_individual_gradients=False,
                               # Will store the ~500 gradients connected to one output node and not just their sum
                               dlg_lr=1,  # learrate of (i)dlg image reconstruction
                               dlg_iterations=100,  # amount of (i)dlg reconstruction iterations
                               log_interval=100000,
                               # Won't store each (i)dlg iteration's images but every n-th iteration's
                               store_composed_image=False,  # storing dlg output images as composed image
                               store_separate_images=False,  # storing dlg output images as seperate images
                               cuda_id=cuda_id,
                               defenses=defenses,
                               differential_privacy=differential_privacy,
                               noise_type=noise_type,
                               noise_multiplier=noise_multiplier,
                               max_norm=beta,
                               compression=compression,
                               threshold=theta,
                               local_iterations=local_iterations,
                               local_training=local_training
                               )

        end = time.time()
        duration = end - start
        print(duration)

    elif job == "visualize":

        ########### Load an existing json an create graphs from it ##########

        run, path = load_jsons(args.dir, args.plot)

        # Visualization Set 0
        if experiment_set == 0:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, labels=["version", "model"])
            # magnitude_check(run, path, gradient_type="original_gradients", group_by="bs")
            # magnitude_check(run, path, gradient_type="adjusted_gradients", group_by="bs")


        # Visualization Set 1
        elif experiment_set == 1:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, width=4.8)

        # Visualization Set 2
        elif experiment_set == 2:
            # default set 2: attack versions
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, width=4.8, labels=["version"])
            # local iteration comparison
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, labels=["local_training"])
            # local iterations & attack versions
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, width=4.8, labels=["version", "local_training"])

        # Visualization Set 1 & 2
        elif experiment_set == 12:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, width=4.8)
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, width=4.8)
            # location="lower right"

        # Visualization Set 3
        elif experiment_set == 3:
            visualize_class_prediction_accuracy_vs_training(run, path, balanced=True, width=4.8)

        # Visualization Set 4
        elif experiment_set == 4:
            visualize_class_prediction_accuracy_vs_training(run, path, balanced=False)
            # visualize_class_prediction_accuracy_vs_training(run, path, balanced=False)

        # Visualization Set 3 & 4
        elif experiment_set == 34:
            visualize_class_prediction_accuracy_vs_training(run, path, balanced=True, width=4.8)
            visualize_class_prediction_accuracy_vs_training(run, path, balanced=False, width=4.8)

        # Visualization Set 5
        elif experiment_set == 5:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, labels="model")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False,
                                                             labels=["version", "model"])

        # Visualization Set 6
        elif experiment_set == 6:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, labels="noise_multiplier")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False,
                                                             labels="noise_multiplier")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, labels="noise_type")

        # Visualization Set 7
        elif experiment_set == 7:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, labels="threshold")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False,
                                                             labels="threshold")

        # Visualization Set 8
        elif experiment_set == 8:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False,
                                                             labels="max_norm")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, labels="noise_type")

        # Visualization Set 9
        elif experiment_set == 9:
            visualize_class_prediction_accuracy_vs_training(run, path, balanced=False,
                                                            width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, balanced=False,
            #                                                 labels=["num_users"], width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, balanced=False,
            #                                                 labels=["local_iterations"], width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, balanced=False,
            #                                                 labels=["noise_multiplier"], width=4.8)

        elif experiment_set == 10:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, width=4.8, labels=["dataset", "local_training"], fontsize=14)

        # Gradient Plots
        if experiment_set == -1:
            y_range = [-330, 300]
            magnitude_check(run, path, gradient_type="original_gradients", balanced="True", dataset="MNIST",
                            version="LLG+", list_bs=[2, 8, 32, 128], group_by="bs", y_range=y_range)
            magnitude_check(run, path, gradient_type="adjusted_gradients", balanced="True", dataset="MNIST",
                            version="LLG+", list_bs=[2, 8, 32, 128], group_by="bs", y_range=y_range)
            heatmap(run, path, gradient_type="original_gradients", balanced="True", dataset="MNIST",
                    list_bs=[2, 8, 32, 128], y_range=y_range)

    else:
        print("Unknown job")

    print("Run finished")


if __name__ == '__main__':
    main()
