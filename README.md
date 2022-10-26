# Fork of the SwAV repository
This repository was forked from the original repository as part of our work to show that "Self-supervised encoders are better transfer learners in remote sensing applications" (pending publication...link forthcoming).

Please see the original repository [here](https://github.com/facebookresearch/swav), and the original paper [here](https://arxiv.org/abs/2006.09882).

This fork was created to make minor changes to the original code so that we could run trials initializing SwAV using the ImageNet weights, versus training from scratch. Our assumption was the pre-training on remote sensing data would be improved if it was first pre-trained on ImageNet, as suggested by this paper [Self-supervised prettraining improves self-supervised pretraining](https://arxiv.org/abs/2103.12718).

This repository only includes minor changes to the original repository. Mainly:
* We made changes to support loading different datasets (i.e., not just ImageNet). We added an argument `task` so that these datasets could be referenced at runtime. There was additionally a small python file added, `src/tasks.py`, that stores the normalization information for each of the tasks.
* We added a boolean argument `initialize_imagenet` that indicated which set of weights should be loaded when running the script. If true, then the model found in the torch hub for the ResNet50 architecture is loaded. Else, the model is randomly initialized. Code is added to `main_swav.py` to handle this initialization.

Please see the `scripts` directory for examples using these extra flags. Otherwise, please feel free to raise an issue should you have a question or notice a bug.
