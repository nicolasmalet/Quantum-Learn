from . import experiments


def main(do_train, do_test, nb_print_train, do_plot_train, do_save):
    experiment = experiments.dudas.instantiate()
    experiment.launch(do_train, do_test, nb_print_train, do_plot_train, do_save)
