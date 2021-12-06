# Label Leakage from Gradients

This framework implements the LLG attack.

## Installation

1. Setup a clean Python `3.7.9`¹ environment with the tool of your choice (conda, venv, etc.).
2. Install required python libraries using: `pip install -r Code/requirements.txt`
3. Initiate and update [aDPtorch](https://github.com/tklab-tud/aDPtorch) submodule: `git submodule init` and `git submodule update`

¹ It is possible that the LLG code runs with newer python versions. However, don't use the most current, as `opacus` and `torchcsprng` tend to have a bit of a delay getting updated to work with newest python and/or torch versions.

## Execution

> TODO: needs updating as well

From main.py you can start an attack like this:


    dataloader = Dataloader()
    experiment(dataloader=dataloader,
               list_datasets=["CIFAR", "MNIST"],
               list_bs=[8, 32],
               list_balanced=[True, False],
               list_versions=["v1", "v2", "random"],
               n=100,
               extent="predict",
               trainsize=100,
               trainsteps=0,
               path=None,
               reconstruction_steps=100
               )

Running on a cluster you might want to comment out :

    from visualize_experiment import *

In order to load jsons it uses a file chooser (from tk_inter) which only works in a desktop environment.

After that the resulting json can be loaded with

run, path = load_json()

See visualize_experiment.py for some possible ways of generating graphs from it.
