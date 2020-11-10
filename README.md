# Label Leakage from Gradients

This framework implements the LLG attack.

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

In order to load jsons it uses a file chooser which only works in a desktop environment. 		   
			   
			   
After that the resulting json can be loaded with
			   
run, path = load_json()

See visualize_experiment.py for some possible ways of generating graphs from it.


 