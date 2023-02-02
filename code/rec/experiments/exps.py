"""Experiments list."""
from rec.experiments.base import run_experiment
from rec.models import lit_model_factory

from yaer.base import experiment


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
        "use_visual_embeddings": True,
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
        "batch_size": 16,
        "grad_steps": 1,
        "max_epochs": 1,
        "scheduler": lambda _: {},
    },
    "runtime_args": {
        "gpus": None,
        "num_workers": 20,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": False,
        "profile": False,
        "checkpoint": None,
        "save_last": False,
        "pdata": 0.02,
        "output_dir": "exp_001",
    }
})
def exp_001():
    """An experiment for testing purposes."""
    run_experiment(model_factory=lit_model_factory)


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
        "use_visual_embeddings": True,
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
        "scheduler": lambda max_epochs: {
            'milestones': [int(p * max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        },
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
        "save_last": False,
        "pdata": 0.02,
        "output_dir": "exp_002"
    }
})
def exp_002():
    """Exp 002 with scheduler."""
    run_experiment(model_factory=lit_model_factory)


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
        "use_visual_embeddings": False,
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
        "scheduler": lambda _: {},
    },
    "runtime_args": {
        "gpus": "0,1",
        "num_workers": 20,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": False,
        "profile": False,
        "checkpoint": None,
        "save_last": False,
        "pdata": 1,
        "output_dir": "exp_003",
    }
})
def exp_003():
    """Exp without using the visual embeddings(by not affecting the masks)."""
    run_experiment(model_factory=lit_model_factory)
