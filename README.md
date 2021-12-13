# Label Leakage from Gradients

This framework implements the Label Leakage from Gradients (LLG) attack, a novel attack to extract ground-truth labels from shared gradients trained with mini-batch stochastic gradient descent for multi-class classification in Federated Learning. LLG is based on a combination of mathematical proofs and heuristics derived empirically. The attack exploits two properties that the gradients of the last layer of a neural network have: (P1) The direction of these gradients indicates whether a label is part of the training batch. (P2) The gradient magnitude can hint towards the number of occurrences of a label in the batch.

### References
[1] [Aidmar Wainakh et al. (2022) User-Level Label Leakage from Gradients in Federated Learning. In PETS 2022.](https://arxiv.org/pdf/2105.09369.pdf)
[2] [Aidmar Wainakh et al. (2021) Label Leakage from Gradients in Distributed Machine Learning. In CCNC 2021.](https://ieeexplore.ieee.org/abstract/document/9369498)

## Installation

1. Setup a clean Python `3.7.9` environment with the tool of your choice (conda, venv, etc.).
2. Install required python libraries using: `pip install -r Code/requirements.txt`
3. Initiate and update [aDPtorch](https://github.com/tklab-tud/aDPtorch) submodule: `git submodule init` and `git submodule update`

It is possible that the LLG code runs with newer python versions. However, don't use the most current, as `opacus` and `torchcsprng` tend to have a bit of a delay getting updated to work with newest python and/or torch versions.

## Execution

1. Choose an experiment from the [table below](#experiment-sets).
2. Prepare the detailed experiment parameters in `main.py` to fit your needs.
3. Execute the experiment:
   `python main.py -s <experiment_set_number> -g <gpu_id_if_avail>`
4. Visualize the dump file(s):
   `python main.py -s <experiment_set_number> -d <path_to_dump_file(s)>`

### Experiment Sets

| set   | description       |
|-------|-------------------|
| 1,2   | batch size (untrained)
| 3,4   | trained model
| 5     | model architecture comparison
| 6     | additive noise (untrained)
| 7     | compression (untrained)
| 8     | differential privacy (untrained)
| 9     | federated training and trained defenses

### Current CLI

```
usage: main.py [-h] [-s SET] [-p PLOT] [-j JOB] [-d DIR] [-g GPU_ID]

Arguments for LLG Experiment

optional arguments:
  -h, --help                    show this help message and exit
  -s SET, --set SET             experiment set (default=2)
  -p PLOT, --plot PLOT          number of files to be ploted (default=None)
  -j JOB, --job JOB             job to execute. either "experiment" or "visualize". (default="experiment")
  -d DIR, --dir DIR             directory or file to plot from. (default=None)
  -g GPU_ID, --gpu_id GPU_ID    cuda_id to use, if available (default=0)
```

## Contributors
- __Aidmar Wainakh__ - _LLG idea, guidance and suggestions during development_
- __Till Müßig__ - _LLG idea, developing LLG and initial experiments as part of his Bachelor’s thesis and a seminar course_
- __Jens Keim__ - _developing advanced experiments, refactoring, current maintainer_

## License

This repository is licensed under the [MIT License](LICENSE.md).

This repo contains a [markdown](LICENSE.md) and a [text](license.txt) version of the license.

In case of any inconstancies refer to the [license's website](https://mit-license.org/).
