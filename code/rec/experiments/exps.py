"""Experiments list."""
from rec.experiments.base import base_experiment

from yaer.base import experiment


@experiment({

})
def exp_001():
    """Default experiment arguments."""
    base_experiment()
