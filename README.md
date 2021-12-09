# Label Leakage from Gradients

This framework implements the LLG attack.

## Installation

1. Setup a clean Python `3.7.9`¹ environment with the tool of your choice (conda, venv, etc.).
2. Install required python libraries using: `pip install -r Code/requirements.txt`
3. Initiate and update [aDPtorch](https://github.com/tklab-tud/aDPtorch) submodule: `git submodule init` and `git submodule update`

¹ It is possible that the LLG code runs with newer python versions. However, don't use the most current, as `opacus` and `torchcsprng` tend to have a bit of a delay getting updated to work with newest python and/or torch versions.

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

## License

This repository is licensed under the [MIT License](LICENSE.md).

This repo contains a [markdown](LICENSE.md) and a [text](license.txt) version of the license.

In case of any inconstancies refer to the [license's website](https://mit-license.org/).
