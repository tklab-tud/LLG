import math
from collections import OrderedDict

from Setting import *
from Graph import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile


# Hypothesis 1
def negativ_value_check(run, path, dataset=None, balanced=None, version="v2", gradient_type="original_gradients"):
    run = run.copy()

    gradient_analysis = {
        "negative":
            {"present": 0,
             "nonpresent": 0},
        "positive":
            {"present": 0,
             "nonpresent": 0},
    }

    meta = run["meta"].copy()
    run.__delitem__("meta")

    for setting in run:
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]):

            for label, gradient in enumerate(run[setting]["prediction_results"][gradient_type]):
                if not isinstance(gradient, list):
                    gradient = [gradient]

                for grad_value in gradient:
                    sign = "positive" if grad_value > 0 else "negative"
                    present = "present" if label in run[setting]["parameter"]["orig_label"] else "nonpresent"
                    gradient_analysis[sign][present] += 1

    result = "Negative + present: {}\tNegative + nonpresent: {}\nPositive + present: {}\tPositive + nonpresent: {} ".format(
        gradient_analysis["negative"]["present"], gradient_analysis["negative"]["nonpresent"],
        gradient_analysis["positive"]["present"], gradient_analysis["positive"]["nonpresent"])

    name = "negative_value_check"
    if dataset is not None:
        name += "_" + dataset
    if balanced is not None:
        name += "_" + str(balanced)

    text_file = open(path + name + "_" + gradient_type + ".txt", "w")
    text_file.write(result)
    text_file.close()


def same_sign_check(run, path, dataset=None, balanced=None):
    run = run.copy()

    gradient_analysis = {
        "positive sum":
            {"positive individual gradient": 0,
             "negative individual gradient": 0},
        "negative sum":
            {"positive individual gradient": 0,
             "negative individual gradient": 0},
    }

    meta = run["meta"].copy()
    run.__delitem__("meta")

    for setting in run:
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset):

            for label, gradient in enumerate(run[setting]["prediction_results"]["original_gradients"]):
                sum_pos = gradient > 0
                for individual_gradient in run[setting]["prediction_results"]["individual_gradients"][label]:
                    ind_pos = individual_gradient > 0
                    gradient_analysis["positive sum" if sum_pos else "negative sum"][
                        "positive individual gradient" if ind_pos else "negative individual gradient"] += 1

    result = "Positive Sum + Positive Individual Gradient: {}\tPositive Sum + Negative Individual Gradient: {}\nNegative Sum + Positive Individual Gradient: {}\tNegative Sum + Negative Individual Gradient: {}" \
        .format(gradient_analysis["positive sum"]["positive individual gradient"],
                gradient_analysis["positive sum"]["negative individual gradient"],
                gradient_analysis["negative sum"]["positive individual gradient"],
                gradient_analysis["negative sum"]["negative individual gradient"])

    text_file = open(path + "same_sign_check" + ".txt", "w")
    text_file.write(result)
    text_file.close()

    # Hypothesis 2


def magnitude_check(run, path, gradient_type="original_gradients", balanced=None, dataset=None, version=None,
                    list_bs=None, trainstep=None, group_by="bs", y_range=None, legend_location="best"):
    if gradient_type == "adjusted_gradients" and run["meta"] == "victim_side":
        print("Adjusted gradients can not be made for a quick run (extend=victim_side)")
        return

    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graphs = []
    for _ in meta["list_bs"]:
        graphs.append(Graph("Label occurrences", "Gradient value", y_range=y_range))

    composed_graph = Graph("Label occurrences", "Gradient value", y_range=y_range)

    print("loading {} from json".format(gradient_type))
    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]

        # get defense
        if run[setting]["parameter"]["compression"]:
            defense = "compression" #+ str(run[setting]["parameter"]["threshold"])
        elif run[setting]["parameter"]["differential_privacy"]:
            defense = "dp" #+ str(run[setting]["parameter"]["noise_multiplier"])
        elif run[setting]["parameter"]["dropout"] != 0.0:
            defense = "dropout" #+ str(run[setting]["parameter"]["dropout"])
        else:
            defense = "none"

        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                trainstep is None or current_meta[5] == str(trainstep)) and (
                version is None or version == current_meta[3]):
            for label, gradient in enumerate(run[setting]["prediction_results"][gradient_type]):
                g = graphs[meta["list_bs"].index(bs)]
                if bs in list_bs:
                    # Colors in the graph will be based on:
                    if group_by == "class":
                        row = "class" + str(label)
                    elif group_by == "defense":
                        row = defense
                    elif group_by == "bs":
                        row = "bs" + str(bs)

                    if not isinstance(gradient, list):
                        gradient = [gradient]

                    for grad_val in gradient:
                        g.add_datapoint(row, grad_val, run[setting]["parameter"]["orig_label"].count(label))
                        composed_graph.add_datapoint(row, grad_val,
                                                     run[setting]["parameter"]["orig_label"].count(label))

    graphs.append(composed_graph)

    filesuffix = meta["list_bs"].copy()
    filesuffix.append("composed")

    for id, graph in enumerate(graphs):
        print("Creating graph: " + str(filesuffix[id]))
        graph.sort()
        graph.plot_scatter(location=legend_location)
        # graph.show()
        name = "Magnitude_BS_{}_{}_".format(filesuffix[id], gradient_type)
        if balanced is not None:
            name += "balanced" if balanced else "unbalanced"
        if dataset is not None:
            name += dataset
        if version is not None:
            name += version
        if group_by == "class":
            name += "_class_colors"
        elif group_by == "defenses":
            name += "_defense_colors"
        elif group_by == "bs":
            name += "_bs_colors"
        name += ".pdf"
        graph.save(path, name)
        # graph.show()
        graph.fig.clf()


def heatmap(run, path, adjusted=True, balanced=None, dataset=None, version=None, list_bs=None):
    if adjusted and run["meta"] == "victim_side":
        print("Adjustment can only be made for a run with extent of prediction/full")
        return

    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()

    gradienttype = "adjusted_gradients" if adjusted else "original_gradients"

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graph = Graph("Label occurrences", "Gradient value")

    print("loading {} from json".format(gradienttype))
    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]) and (
                bs in list_bs):
            for label, gradient in enumerate(run[setting]["prediction_results"][gradienttype]):
                graph.add_datapoint("bs" + str(bs), gradient, run[setting]["parameter"]["orig_label"].count(label))

    print("Creating graph")
    graph.sort()
    graph.plot_heatmap()
    # graph.show()
    name = "Heatmap_"
    if balanced is not None:
        name += "balanced" if balanced else "unbalanced"
    if adjusted is not None:
        name += "adjusted" if adjusted else "original"
    name += ".pdf"
    graph.save(path, name)
    graph.fig.clf()


def pearson_check(run, path, balanced=None, dataset=None, version=None, list_bs=None):
    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graph = Graph("Label occurrences", "Gradient value")
    result_string = ""
    result_list_original = []
    result_list_adjusted = []

    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]) and (
                bs in list_bs):
            l_x = []
            l_y_original = []
            l_y_adjusted = []
            for label in range(run[setting]["parameter"]["num_classes"]):
                l_x.append(run[setting]["parameter"]["orig_label"].count(label))
                l_y_original.append(run[setting]["prediction_results"]["original_gradients"][label])
                l_y_adjusted.append(run[setting]["prediction_results"]["adjusted_gradients"][label])

            pearson_r_original = scipy.stats.pearsonr(l_x, l_y_original)
            pearson_r_adjusted = scipy.stats.pearsonr(l_x, l_y_adjusted)

            result_string += "{}: pearson_r_original = {:.5f}\t pearson_r_adjusted = {:.5f}\n". \
                format(current_meta, pearson_r_original[0], pearson_r_adjusted[0])
            result_list_original.append(pearson_r_original[0])
            result_list_adjusted.append(pearson_r_adjusted[0])

    result_string = "MEAN: pearson_r_original = {:.5f}\t pearson_r_adjusted = {:.5f}\n\n".format(
        np.mean(result_list_original), np.mean(result_list_adjusted)) + result_string

    text_file = open(path + "pearson_check.txt", "w")
    text_file.write(result_string)
    text_file.close()


# Experiment 1.1
def visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=None, dataset=None, version=None):
    run = run.copy()

    graph = Graph("Batch Size", "Label extraction accuracy")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Prediction Acc
    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):
            label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
                current_meta[3] == "random" else "LLG*" if current_meta[3] in ["v3-one",  "v3-zero", "v3-random"] else "DLG" if \
                current_meta[3] == "dlg" else "iDLG" if current_meta[3] == "idlg" else "?"
                # current_meta[3] == "random" else "LLG-ONE" if current_meta[3] == "v3-one" else "LLG-ZERO" if \
                # current_meta[3] == "v3-zero" else "LLG-RANDOM" if current_meta[3] == "v3-random" else "DLG" if \
            # label += " "
            # label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

            graph.add_datapoint(label, run[run_name]["prediction_results"]["accuracy"], str(current_meta[1]))

    graph.plot_line()

    # graph.show()

    name = "class_prediction_accuracy_vs_batchsize_"
    if balanced is not None:
        name += "_balanced" if balanced else "_unbalanced"
    if dataset is not None:
        name += "_" + dataset
    name += ".pdf"

    graph.save(path, name)


def visualize_hellinger_vs_batchsize(run, path, balanced=None, dataset=None, version=None):
    run = run.copy()

    graph = Graph("Batch Size", "Hellinger distance")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Prediction Acc
    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):
            label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
                current_meta[3] == "random" else "LLG-ONE" if current_meta[3] == "v3-one" else "LLG-ZERO" if \
                current_meta[3] == "v3-zero" else "LLG-RANDOM" if current_meta[3] == "v3-random" else "?"
            label += " "
            label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

            # calculate hellinger distance between extraction and ground truth
            list_of_squares = []
            p = probability_distribution(run[run_name]["prediction_results"]["prediction"],
                                         run[run_name]["parameter"]["num_classes"])
            g = probability_distribution(run[run_name]["parameter"]["orig_label"],
                                         run[run_name]["parameter"]["num_classes"])
            for p_i, g_i in zip(p, g):
                # caluclate the square of the difference of ith distr elements
                s = (math.sqrt(p_i) - math.sqrt(g_i)) ** 2

                # append
                list_of_squares.append(s)

            # calculate sum of squares
            sosq = sum(list_of_squares)

            hellinger = sosq / math.sqrt(2)

            graph.add_datapoint(label, hellinger, str(current_meta[1]))

    graph.plot_line()

    # graph.show()

    graph.save(path, "hellinger_vs_batchsize.pdf")


# Experiment 1.2
def visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, balanced=None, dataset=None, version=None):
    run = run.copy()

    graph = Graph("Batch Size", "Flawless label extraction share")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    points = {}

    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")
        label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
            current_meta[3] == "random" else "LLG*" if current_meta[3] in ["v3-one",  "v3-zero", "v3-random"] else "?"
            # current_meta[3] == "random" else "LLG-ONE" if current_meta[3] == "v3-one" else "LLG-ZERO" if \
            # current_meta[3] == "v3-zero" else "LLG-RANDOM" if current_meta[3] == "v3-random" else "?"
        # label += " "
        # label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):

            if run[run_name]["prediction_results"]["accuracy"] == 1.0:
                graph.add_datapoint(label, 1, current_meta[1])
            else:
                graph.add_datapoint(label, 0, current_meta[1])

    graph.plot_line()
    # graph.show()

    name = "flawless_class_prediction_accuracy_vs_batchsize"
    if balanced is not None:
        name += "_balanced" if balanced else "_unbalanced"
    if dataset is not None:
        name += "_" + dataset
    name += ".pdf"

    graph.save(path, name)

    return graph


# Experiment 2
def visualize_class_prediction_accuracy_vs_training(run, path, balanced=None, dataset=None, version=None, list_bs=None,
                                                    train_step_stop=None):
    run = run.copy()

    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graph = Graph("x{} Iterations".format(meta["trainsize"]), "Label extraction accuracy", "Model accuracy (%)")

    data2 = []

    # Prediction Acc
    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]) and (
                train_step_stop is None or train_step_stop >= int(current_meta[5])):
            label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
                current_meta[3] == "random" else "LLG-ONE" if current_meta[3] == "v3-one" else "LLG-ZERO" if \
                current_meta[3] == "v3-zero" else "LLG-RANDOM" if current_meta[3] == "v3-random" else "?"
            label += " "
            label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

            graph.add_datapoint(label, run[run_name]["prediction_results"]["accuracy"], str(current_meta[5]))
            data2.append(["model accuracy", run[run_name]["parameter"]["test_acc"], str(current_meta[5])])

    graph.data.append(["model accuracy", 0, str(0)])
    graph.plot_line(location="center right", move=(1, 0.4), skip_x_ticks=True)
    graph.data = data2
    graph.plot_line(True, legend=False, skip_x_ticks=True)

    # graph.show()

    name = "class_prediction_accuracy_vs_training"
    if balanced is not None:
        name += "_balanced" if balanced else "_unbalanced"
    if dataset is not None:
        name += "_" + dataset
    name += ".pdf"

    graph.save(path, name)


# Experiment 3: Good Fidelity
# Instead of evaluating class prediction accuracy this experiment performs full recreation attacks.
# It will evaluate the image similarity by plotting the percentage of samples that reach a mse below a threshold.
# The threshold is plotted to the x axis.

def visualize_good_fidelity(run, path, fidelitysteps, bs, balanced):
    run = run.copy()

    graph = Graph("Fidelity Score", "Percentage of Samples")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # prepare fidelity
    fidelity = np.zeros((len(fidelitysteps), len(meta["list_versions"])))

    # go through run

    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")

        if int(current_meta[1]) == bs and current_meta[2] == str(balanced):
            for i_step, step in enumerate(fidelitysteps):
                for mse in run[run_name]["attack_results"]["mses"][-1]:
                    if mse < step:
                        fidelity[i_step][meta["list_versions"].index(current_meta[3])] += 1

    length = bs * meta["n"]

    for step, row in enumerate(fidelity):
        for version, value in enumerate(row):
            # Name in the graph
            label = meta["list_versions"][version]
            label = "LLG" if label == "v1" else "LLG+" if label == "v2" else "Random" if label == "random" \
                else "iDLG" if label == "idlg" else "DLG" if label == "dlg" else "?"
            label += " "
            label += "(IID)" if balanced else "(non-IID)"

            y = fidelity[step][version] / length
            x = str(fidelitysteps[step])

            graph.add_datapoint(label, y, x)

    graph.plot_line()
    # graph.show()
    graph.save(path, "good_fidelity.pdf")


def get_meta(run, cut_meta=False):

    meta = run["meta"].copy()
    if cut_meta:
        run.__delitem__("meta")

    return run, meta

def compare_meta(meta1, meta2):
    for key in meta1.keys():
        elem1 = meta1[key]
        elem2 = meta2[key]
        if elem1 != elem2:
            print(key, ":", elem1, "or", elem2)

def merge_runs(run_meta, run):

    run_meta.update(run)

    return run_meta

def load_json():
    Tk().withdraw()
    filename = askopenfilename(initialdir="./results", defaultextension='.json',
                               filetypes=[('Json', '*.json')])

    with open(filename) as f:
        dump = OrderedDict(json.load(f))

    path = os.path.split(f.name)[0]

    return dump, path + "/"


def probability_distribution(v, n):
    distribution = np.zeros(n)
    for x in v:
        distribution[x] += 1

    distribution = np.divide(distribution, sum(distribution))

    return distribution
