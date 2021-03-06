import math
from collections import OrderedDict
import os, json

from Setting import *
from Graph import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile

fontsize = 16


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
                    list_bs=None, trainstep=None, group_by="bs", y_range=None, legend_location="best", width=6.4):
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
        graphs.append(Graph("Label occurrences", "Gradient value", y_range=y_range, fontsize=fontsize, width=width))

    composed_graph = Graph("Label occurrences", "Gradient value", y_range=y_range, fontsize=fontsize, width=width)

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
                        occurrences = sum(x.count(label) for x in run[setting]["parameter"]["orig_label"])
                        g.add_datapoint(row, grad_val, occurrences)
                        composed_graph.add_datapoint(row, grad_val,occurrences)

    graphs.append(composed_graph)

    filesuffix = meta["list_bs"].copy()
    filesuffix.append("composed")

    for id, graph in enumerate(graphs):
        print("Creating graph: " + str(filesuffix[id]))
        graph.sort()
        graph.plot_scatter(location=legend_location)
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
        graph.fig.clf()


def heatmap(run, path, gradient_type="original_gradients", balanced=None, dataset=None, version=None, list_bs=None, y_range=None, trainstep=None, width=6.4):

    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()


    meta = run["meta"].copy()
    run.__delitem__("meta")

    graph = Graph("Label occurrences", "Gradient value", y_range=y_range, fontsize=fontsize, width=width)

    print("loading {} from json".format(gradient_type))
    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                trainstep is None or current_meta[5] == str(trainstep)) and (
                version is None or version == current_meta[3]):
            for label, gradient in enumerate(run[setting]["prediction_results"][gradient_type]):
                graph.add_datapoint("bs" + str(bs), gradient, run[setting]["parameter"]["orig_label"].count(label))

    print("Creating graph")
    graph.sort()
    graph.plot_heatmap()
    name = "Heatmap_BS_{}_".format( gradient_type)
    if list_bs is not None:
        name += str(list_bs)
    if balanced is not None:
        name += "balanced" if balanced else "unbalanced"
    if dataset is not None:
        name += dataset
    if version is not None:
        name += version
    name += ".pdf"
    graph.save(path, name)
    graph.fig.clf()


def pearson_check(run, path, balanced=None, dataset=None, version=None, list_bs=None, width=6.4):
    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graph = Graph("Label occurrences", "Gradient value", fontsize=fontsize, width=width)
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
def visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=None, dataset=None, version=None, labels="", width=6.4, location="best", y_range=[0,105], fontsize=16):
    run = run.copy()

    graph = Graph("Batch size (log2)", "Attack success rate (%)", y_range=y_range, fontsize=fontsize, width=width)

    if not isinstance(run, list):
        runs = [run]
    else:
        runs = run

    for run in runs:
        # Prediction Acc
        for id, run_name in enumerate(run):
            if run_name == "meta":
                continue
            current_meta = run_name.split("_")
            if dataset is None:
                dataset = current_meta[0]
            if (balanced is None or current_meta[2] == str(balanced)) and (
                    dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):
                label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
                    current_meta[3] == "random" else "LLG*" if "llg*" in current_meta[3] or "v3" in current_meta[3] else "DLG" if \
                    current_meta[3] == "dlg" else "iDLG" if current_meta[3] == "idlg" else current_meta[3]

                # labels need to be one of the attack parameters
                # e.g. "model", "threshold", "noise_multiplier"

                merged_label = ""
                if not isinstance(labels, list): labels = [labels]
                for l in labels:
                    if l != "":
                        label = run[run_name]["parameter"][l]
                    if l == "version":
                        if label.lower() == "llg+":
                            label = "LLG+"
                        elif label.lower() =="dlg":
                            label = "DLG"
                        elif label.lower() == "llg":
                            label = "LLG"
                        elif "llg*" in label.lower():
                            label = "LLG*"
                        elif label.lower() == "random":
                            label = "Random"

                    if l == "model":
                        if label == "MLP":
                            label = "FCNN"
                    if l == "threshold":
                        if run[run_name]["parameter"]["version"] == "random":
                            label = "Random"
                        elif run[run_name]["parameter"]["compression"] == False:
                            label = "??=0%"
                        elif label == 0.1:
                            label = "??=10%"
                        elif label == 0.2:
                            label = "??=20%"
                        elif label == 0.4:
                            label = "??=40%"
                        elif label == 0.8:
                            label = "??=80%"
                        elif label == 0.0:
                            label = "??=0%"
                    if l == "noise_multiplier":
                        if run[run_name]["parameter"]["version"] == "random":
                            label = "Random"
                        elif run[run_name]["parameter"]["differential_privacy"] == False:
                            label = "No noise"
                        elif label == 0.0:
                            label = "No noise"
                        else:
                            label = "???? = " + str(label)
                    if l == "max_norm":
                        if run[run_name]["parameter"]["version"] == "random":
                            label = "Random"
                        elif run[run_name]["parameter"]["differential_privacy"] == False:
                            label = "No noise"
                        elif run[run_name]["parameter"]["noise_multiplier"] == 0.0:
                            label = "No noise"
                        elif label == 0.0:
                            label = "?? = 0"
                        elif label == None:
                            label = "?? = ???"
                        else:
                            label = "?? = " + str(int(label))
                    if l == "local_iterations":
                        label = "FedAvg-Iterations: "+str(run[run_name]["parameter"]["local_iterations"])
                    if l == "local_training":
                        if run[run_name]["parameter"]["local_training"]:
                            label = "FedAvg (E={})".format(run[run_name]["parameter"]["local_iterations"])
                        else:
                            label = "FedSGD"



                    merged_label = merged_label + str(label) + ", "
                merged_label = merged_label[:-2]


                graph.add_datapoint(merged_label, run[run_name]["prediction_results"]["accuracy"]*100, current_meta[1])

    graph.plot_line(location=location, legend=False)

    name = "class_prediction_accuracy_vs_batchsize_"
    if balanced is not None:
        name += "_balanced" if balanced else "_unbalanced"
    if dataset is not None:
        name += "_" + dataset
    name += ".pdf"

    graph.save(path, name)


def visualize_hellinger_vs_batchsize(run, path, balanced=None, dataset=None, version=None, width=6.4):
    run = run.copy()

    graph = Graph("Batch Size", "Hellinger distance", fontsize=fontsize, width=width)
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

    graph.save(path, "hellinger_vs_batchsize.pdf")


# Experiment 1.2
def visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, balanced=None, dataset=None, version=None, width=6.4):
    run = run.copy()

    graph = Graph("Batch Size", "Flawless label extraction share", fontsize=fontsize, width=width)
    meta = run["meta"].copy()
    run.__delitem__("meta")

    points = {}

    for id, run_name in enumerate(run):
        current_meta = run_name.split("_")
        label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
            current_meta[3] == "random" else "LLG*" if current_meta[3] in ["v3-one",  "v3-zero", "v3-random"] else "?"

        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):

            if run[run_name]["prediction_results"]["accuracy"] == 1.0:
                graph.add_datapoint(label, 1, current_meta[1])
            else:
                graph.add_datapoint(label, 0, current_meta[1])

    graph.plot_line()

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
                                                    train_step_stop=None, labels="", model_id=1, width=6.4, location="best", fontsize=16):
    run = run.copy()

    if not isinstance(run, list):
        runs = [run]
    else:
        runs = run

    graph = Graph("Iterations", "Attack success rate (%)", "Model accuracy (%)", y_range=[0,105], fontsize=fontsize, width=width) # (x100)

    data2 = []

    model_accs = {}

    for id, run in enumerate(runs):
        # Prediction Acc
        for run_name in run:
            if run_name == "meta":
                meta = run["meta"].copy()
                continue
            run_name = run_name.replace("victim_side", "victimside")
            current_meta = run_name.split("_")
            run_name = run_name.replace("victimside", "victim_side")
            if current_meta[4] == "victimside":
                current_meta[4] = "victim_side"
            if dataset is None:
                dataset = current_meta[0]
            if (balanced is None or current_meta[2] == str(balanced)) and (
                    dataset is None or current_meta[0] == dataset) and (
                    version is None or version == current_meta[3]) and (
                    train_step_stop is None or train_step_stop >= int(current_meta[5])):
                label = "LLG" if current_meta[3] == "v1" else "LLG+" if current_meta[3] == "v2" else "Random" if \
                    current_meta[3] == "random" else "LLG*" if "llg*" in current_meta[3] or "v3" in current_meta[3] else "DLG" if \
                    current_meta[3] == "dlg" else "iDLG" if current_meta[3] == "idlg" else "?"

                # labels need to be one of the attack parameters
                # e.g. "model", "threshold", "noise_multiplier"
                merged_label = ""
                if not isinstance(labels, list): labels = [labels]
                for l in labels:
                    if l != "":
                        label = run[run_name]["parameter"][l]
                    if l == "version":
                        if run[run_name]["parameter"]["federated"]:
                            label = "FL"
                        elif label.lower() == "llg+":
                            label = "LLG+"
                        elif label.lower() =="dlg":
                            label = "DLG"
                        elif label.lower() == "llg":
                            label = "LLG"
                        elif "llg*" in label.lower():
                            label = "LLG*"
                        elif label.lower() == "random":
                            label = "Random"
                    if l == "num_users":
                        label = "N = " + str(run[run_name]["parameter"]["num_users"])
                        if not run[run_name]["parameter"]["federated"]:
                            label = "LLG+"
                    if l == "model":
                        if label == "MLP":
                            label = "FCNN"
                    if l == "noise_multiplier":
                        if run[run_name]["parameter"]["version"] == "random":
                            label = "Random"
                        elif run[run_name]["parameter"]["differential_privacy"] == False:
                            label = "No noise"
                        elif label == 0.0:
                            label = "No noise"
                        else:
                            label = "???? = " + str(label)
                    if l == "local_iterations":
                        label = "FedAvg-Iterations: " + str(run[run_name]["parameter"]["local_iterations"])
                    if l == "local_training":
                        if run[run_name]["parameter"]["local_training"]:
                            label = "FedAvg({} iterations)".format(run[run_name]["parameter"]["local_iterations"])
                        else:
                            label = "FedSGD"


                    merged_label = merged_label + str(label) + ", "
                merged_label = merged_label[:-2]

                x_tick_name = int(int(current_meta[5])*meta["trainsize"]*run[run_name]["parameter"]["local_iterations"])

                # if "DLG" in label:
                #     x_tick_name = x_tick_name*10

                graph.add_datapoint(merged_label, run[run_name]["prediction_results"]["accuracy"]*100, x_tick_name)

                acc = float(run[run_name]["parameter"]["test_acc"])
                if x_tick_name in model_accs.keys():
                    model_accs.update({x_tick_name: model_accs[x_tick_name].__add__([acc])})
                else:
                   model_accs.update({x_tick_name: [acc]})

    data2 = [["Model", np.average(val) , key] for key, val in model_accs.items()]
    print("list: ", model_accs[list(model_accs)[-1]])
    print("average: ", np.average(model_accs[list(model_accs)[-1]]))

    if len(data2) >= 1000:
        data2 = [val for i, val in enumerate(data2) if i % 10 == 0]

    graph.data.append(["Model", 0, 0])
    graph.plot_line(location=location, skip_x_ticks=True, useMathText=True, legend=False)
    graph.data = data2
    graph.plot_line(True, legend=False, skip_x_ticks=True, location=location, useMathText=True)

    name = "class_prediction_accuracy_vs_training"
    if balanced is not None:
        name += "_balanced" if balanced else "_unbalanced"
    if dataset is not None:
        name += "_" + dataset
    # name += "_" + str(model_id)
    name += ".pdf"

    graph.save(path, name)


# Experiment 3: Good Fidelity
# Instead of evaluating class prediction accuracy this experiment performs full recreation attacks.
# It will evaluate the image similarity by plotting the percentage of samples that reach a mse below a threshold.
# The threshold is plotted to the x axis.
def visualize_good_fidelity(run, path, fidelitysteps, bs, balanced, width=6.4):
    run = run.copy()

    graph = Graph("Fidelity Score", "Percentage of Samples", fontsize=fontsize, width=width)
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
                else "iDLG" if label == "idlg" else "LLG*" if "llg*" in current_meta[3] or "v3" in current_meta[3] \
                else "DLG" if label == "dlg" else label
            label += " "
            label += "(IID)" if balanced else "(non-IID)"

            y = fidelity[step][version] / length
            x = str(fidelitysteps[step])

            graph.add_datapoint(label, y, x)

    graph.plot_line()
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

def append_runs(run_meta, run):

    if not isinstance(run_meta, list):
        run_meta = [run_meta]
    run_meta.append(run)

    return run_meta

def merge_runs(run_meta, run):

    run_meta.update(run)

    return run_meta

def load_jsons(path_to_json, num_files=1):
    if path_to_json == None:
        return load_jsons_old(num_files)
    if "," in path_to_json:
        paths_to_jsons = path_to_json.split(",")
        json_files = []
        for path_to_json in paths_to_jsons:
            json_files.extend(fetch_jsons(path_to_json))
    else:
        json_files = fetch_jsons(path_to_json)
    return parse_jsons(json_files)

def fetch_jsons(path_to_json):
    if ".json" in path_to_json:
        json_files = [path_to_json]
    else:
        json_files = []
        for folder in os.listdir(path_to_json):
            if ".json" in folder:
                json_files = [path_to_json+"/"+folder]
                continue
            if not os.path.isdir(path_to_json+"/"+folder):
                continue
            for file_name in os.listdir(path_to_json+"/"+folder):
                if ".json" in file_name:
                    file_path = path_to_json+"/"+folder+"/"+file_name
                    json_files.append(file_path)
    return json_files

def parse_jsons(json_files):
    runs = []
    path = None
    meta = None
    for json_file in json_files:
        run, path_tmp = open_json(json_file)
        _, meta_tmp = get_meta(run)
        if meta == None:
            meta = meta_tmp
        if path == None:
            path = path_tmp
        compare_meta(meta, meta_tmp)
        runs = append_runs(runs, run)
    return runs, path

def load_jsons_old(num_files):
    run, path = load_json()
    _, meta = get_meta(run)

    for i in range(num_files - 1):
        run2, path2 = load_json()
        run2, meta2 = get_meta(run2, cut_meta=True)

        compare_meta(meta, meta2)

        run = append_runs(run, run2)
    return run, path

def load_json():
    Tk().withdraw()
    filename = askopenfilename(initialdir="./results", defaultextension='.json',
                               filetypes=[('Json', '*.json')])

    return open_json(filename)

def open_json(filename):
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
