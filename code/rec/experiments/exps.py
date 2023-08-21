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
        "force_ddp": False,
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
        "output_dir": "exp_002",
        "get_sample": True
    }
})
def exp_002():
    """Exp 002 with scheduler parameter defined. Useless experiment."""
    run_experiment(model_factory=lit_model_factory)


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 0,
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
        "mu": 0.0,
    },
    "trainer_args": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "batch_size": 12,
        "grad_steps": 4,
        "max_epochs": 90,
        "scheduler": lambda max_epochs: {
            'milestones': [int(p * max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        }
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
        "checkpoint": True,
        "save_last": False,
        "pdata": 1.0,
        "output_dir": "exp_003",
        "get_sample": False
    }
})
def exp_003():
    """Define an exp that is the same as the paper (referit->baseline).

    Is the 20211220_201458 prefix folder name from paper's google drive versioned models.
    The idea for this experiment is to compare its results with the reported ones in the paper.
    """
    run_experiment(model_factory=lit_model_factory)


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 0,
        "use_visual_embeddings": False,
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
        "mu": 0.0,
    },
    "trainer_args": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "batch_size": 12,
        "grad_steps": 4,
        "max_epochs": 90,
        "scheduler": lambda max_epochs: {
            'milestones': [int(p * max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        }
    },
    "runtime_args": {
        "gpus": "0,1,2",
        "num_workers": 20,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": False,
        "profile": False,
        "checkpoint": True,
        "save_last": True,
        "pdata": 1.0,
        "output_dir": "exp_004",
        "get_sample": False
    }
})
def exp_004():
    """Define an exp that is the same as the paper (referit->baseline), but shuting down the visual embeddings."""
    run_experiment(model_factory=lit_model_factory)


@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 0,
        "use_visual_embeddings": False,
        "use_visual_pos_embeddings": False,
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
        "mu": 0.0,
    },
    "trainer_args": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "batch_size": 12,
        "grad_steps": 4,
        "max_epochs": 90,
        "scheduler": lambda max_epochs: {
            'milestones': [int(p * max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        }
    },
    "runtime_args": {
        "gpus": "0,1,2",
        "num_workers": 20,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": False,
        "profile": False,
        "checkpoint": True,
        "save_last": True,
        "pdata": 1.0,
        "output_dir": "exp_005",
        "get_sample": False
    }
})
def exp_005():
    """Define an exp that is the same as exp_004, but shuting down the visual positional embeddings."""
    run_experiment(model_factory=lit_model_factory)
