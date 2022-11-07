"""Experiments list."""
from rec.experiments.base import base_experiment

from yaer.base import experiment


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
    },
    "data_args": {
        "dataset": "refcoco",
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
        "batch_size": 16,
        "grad_steps": 1,
        "max_epochs": 1,
        "scheduler": False,
    },
    "runtime_args": {
        "gpus": "0",
        "num_workers": 20,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": True,
        "early_stopping": False,
        "amp": False,
        "force_ddp": False,
        "profile": False,
        "checkpoint": None,
        "save_last": False
    }
})
def exp_001():
    """Default experiment arguments."""
    base_experiment()
