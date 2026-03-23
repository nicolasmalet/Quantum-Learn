from .experiments import EXPERIMENTS


def main(do_train, do_test, nb_print_train, do_plot_train, do_save):
    experiment = EXPERIMENTS.photon.instantiate()
    experiment.launch(do_train, do_test, nb_print_train, do_plot_train, do_save)
