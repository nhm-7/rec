# Using YAER for Experiment Versioning

## Description and use cases
Most scientists aim to make their experiments reproducible under certain initial conditions, or at least produce identical outputs for identical inputs. In this work, we not only ran experiments using and modifying the baseline model described in Chapter 2, but we also leveraged a public library available on GitHub called **Yet Another Experiment Runner (YAER)**.

Technically, YAER is a set of Python decorators based on the idea that an experiment is simply a function with arguments. This library allows us to centralize all experiments in a single file, making most of the arguments for a given experiment explicitly defined in a Python file. This way, the reproducibility of an experiment becomes strictly tied to the version of the code itself.

## Components

YAER has two main high-level components:
`experiment_component` and `experiment`.

These decorators allow us to label functions related to our experiments. If an experiment consists of a set of components that are related to each other, those components are tagged with the `experiment_component` decorator. On the other hand, the main experiment function that centralizes the entire configuration is tagged with the `experiment` decorator.

This decorator defines a dictionary where the keys are the experiment’s arguments and the values are the corresponding parameters passed at runtime when running that experiment. You can check the implementation of both components directly in the [YAER repository](https://github.com/arielrossanigo/yaer/tree/master).

## How we used it in this work
In this project, all experiments were defined in the `exps.py` file. For example, the experiment `exp_001` was defined as follows:

```python
@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
        "use_visual_embeddings": True,
        "use_visual_pos_embeddings": True,
        "visual_pos_emb": {
            "name": "learned_pos_emb_2d",
            "args": {
                "embedding_dim": 256,
            },
        },
    },
    "data_args": {
        "dataset": "refclef",
        "max_length": 32,
        "input_size": 512,
    },
    "loss_args": {
        "beta": 0.1,
        "gamma": 0.1,
        "mu": 0.1,
    },
    "trainer_args": {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "batch_size": 4,
        "grad_steps": 1,
        "max_epochs": 1,
        "scheduler": lambda _: {},
    },
    "runtime_args": {
        "gpus": None,
        "num_workers": 8,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": True,
        "profile": False,
        "checkpoint": None,
        "save_last": False,
        "pdata": 0.34,
        "output_dir": "exp_001",
        "get_sample": True
    }
})
def exp_001():
    """An experiment for testing purposes (refactors, etc)."""
    run_experiment(model_factory=lit_model_factory)
```

As we can see, the `experiment` decorator defines all the arguments that `exp_001` will use at runtime. These argument sets are primarily consumed in `base.py`, but many of them are cascaded and updated in lower-level files such as `models.py`, particularly in the `lit_model_factory` function. In all these files, the `experiment_component` decorator is used, signaling YAER that arguments should be updated during the execution of the experiment.
