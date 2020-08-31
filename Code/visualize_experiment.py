from collections import OrderedDict

from Setting import *
from Graph import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile


# Hypothesis 1
def negativ_value_check(run, path):
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
        for label, gradient in enumerate(run[setting]["prediction_results"]["original_gradients"]):
            sign = "positive" if gradient > 0 else "negative"
            present = "present" if label in run[setting]["parameter"]["orig_label"] else "nonpresent"
            gradient_analysis[sign][present] += 1

    result = "Negative + present: {}\tNegative + nonpresent: {}\nPositive + present: {}\tPositive + nonpresent: {} ".format(
        gradient_analysis["negative"]["present"], gradient_analysis["negative"]["nonpresent"],
        gradient_analysis["positive"]["present"], gradient_analysis["positive"]["nonpresent"])

    text_file = open(path + "negative_value_check.txt", "w")
    text_file.write(result)
    text_file.close()

    # Hypothesis 2


def magnitude_check(run, path, adjusted=True, balanced=None, dataset=None, version=None, list_bs=None):
    if adjusted and run["meta"] == "victim_side":
        print("Adjustment can only be made for a run with extent of prediction/full")
        return

    if list_bs is None:
        list_bs = run["meta"]["list_bs"]

    run = run.copy()

    gradienttype = "adjusted_gradients" if adjusted else "original_gradients"

    meta = run["meta"].copy()
    run.__delitem__("meta")

    graphs = []
    for _ in meta["list_bs"]:
        graphs.append(Graph("Occurrences", "Gradient value"))

    composed_graph = Graph("Occurrences", "Gradient value")

    print("loading {} from json".format(gradienttype))
    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]):
            for label, gradient in enumerate(run[setting]["prediction_results"][gradienttype]):
                g = graphs[meta["list_bs"].index(bs)]
                if bs in list_bs:
                    g.add_datapoint(bs, gradient, run[setting]["parameter"]["orig_label"].count(label))
                    composed_graph.add_datapoint(bs, gradient, run[setting]["parameter"]["orig_label"].count(label))



    graphs.append(composed_graph)

    filesuffix = meta["list_bs"].copy()
    filesuffix.append("composed")

    for id, graph in enumerate(graphs):
        print("Creating graph: " + str(filesuffix[id]))
        graph.sort()
        graph.plot_scatter()
        # graph.show()
        name = "Magnitude_BS_{}_{}_".format(filesuffix[id], gradienttype)
        if balanced is not None:
            name += "balanced" if balanced else "unbalanced"
        name += ".pdf"
        graph.save(path, name)
        graph.show()
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

    graph = Graph("Occurrences", "Gradient value")

    print("loading {} from json".format(gradienttype))
    for i, setting in enumerate(run):
        bs = run[setting]["parameter"]["batch_size"]
        current_meta = setting.split("_")
        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (
                version is None or version == current_meta[3]) and (
                bs in list_bs):
            for label, gradient in enumerate(run[setting]["prediction_results"][gradienttype]):
                graph.add_datapoint(bs, gradient, run[setting]["parameter"]["orig_label"].count(label))



    print("Creating graph")
    graph.sort()
    graph.plot_heatmap()
    graph.show()
    name = "Heatmap_"
    if balanced is not None:
        name += "balanced" if balanced else "unbalanced"
    if adjusted is not None:
        name += "adjusted" if adjusted else "original"
    name += ".pdf"
    graph.save(path, name)
    graph.fig.clf()


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
                current_meta[3] == "random" else "?"
            label += " "
            label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

            graph.add_datapoint(label, run[run_name]["prediction_results"]["accuracy"], str(current_meta[1]))

    graph.plot_line()

    graph.show()

    graph.save(path, "class_prediction_accuracy_vs_batchsize.pdf")


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
            current_meta[3] == "random" else "?"
        label += " "
        label += "(IID)" if current_meta[2] == "True" else "(non-IID)" if current_meta[2] == "False" else "?"

        if (balanced is None or current_meta[2] == str(balanced)) and (
                dataset is None or current_meta[0] == dataset) and (version is None or version == current_meta[3]):

            if run[run_name]["prediction_results"]["accuracy"] == 1.0:
                graph.add_datapoint(label, 1, current_meta[1])
            else:
                graph.add_datapoint(label, 0, current_meta[1])

    graph.plot_line()
    graph.show()
    graph.save(path, "flawles_class_prediction_accuracy_vs_batchsize.pdf")

    return graph


# Experiment 2
def visualize_class_prediction_accuracy_vs_training(run, path):
    run = run.copy()

    graph = Graph("Train Samples", "Label extraction Accuracy", "Test Acc")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Prediction Acc
    for id, run_name in enumerate(run):
        i = id % meta["n"]
        trainstep = id // meta["n"]
        graph.add_datapoint(meta["version"], run[run_name]["prediction_results"]["accuracy"], trainstep)

    graph.plot_line( alt_ax=False)
    graph.data = []

    # Test Acc
    for id, run_name in enumerate(run):
        i = id % meta["n"]
        trainstep = id // meta["n"]
        graph.add_datapoint("test_acc", run[run_name]["parameter"]["test_acc"], trainstep)

    graph.plot_line( alt_ax=True)

    graph.show()
    graph.save(path, "class_prediction_accuracy_vs_training.pdf")


# Experiment 3: Good Fidelity
# Instead of evaluating class prediction accuracy this experiment performs full recreation attacks.
# It will evaluate the image similarity by plotting the percentage of samples that reach a mse below a threshold.
# The threshold is plotted to the x axis.

def visualize_good_fidelity(run, path):
    run = run.copy()

    graph = Graph("Fidelity Score", "Percentage of Samples")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # prepare fidelity
    fidelity = {}
    for strat in meta["strats"]:
        fidelity.update({strat: {}})
        for step in meta["steps"]:
            fidelity[strat].update({step: 0})

    # go through run
    for id, run_name in enumerate(run):

        for step in meta["steps"]:
            for mse in run[run_name]["attack_results"]["mses"][-1]:
                if mse < step:
                    fidelity[run[run_name]["parameter"]["prediction"]][step] += 1

    length = meta["bs"] * meta["n"]

    for i, strat in enumerate(fidelity):
        for step in fidelity[strat]:
            graph.add_datapoint(strat, fidelity[strat][step] / length, str(step))
        style = "solid" if i == 1 else "--"
        graph.plot_line(style=style)
        graph.data = []

    graph.show()
    graph.save(path, "good_fidelity.pdf")


def load_json():
    Tk().withdraw()
    filename = askopenfilename(initialdir="./results", defaultextension='.json',
                               filetypes=[('Json', '*.json')])

    with open(filename) as f:
        dump = OrderedDict(json.load(f))

    path = os.path.split(f.name)[0]

    return dump, path + "/"
