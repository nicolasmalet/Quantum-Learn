from . import experiments


def main():
    experiment_config = experiments.photon
    experiment_config.summary()
    experiment = experiment_config.instantiate()
