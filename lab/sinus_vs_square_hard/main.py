from lab.sinus_vs_square_hard.experiments import EXPERIMENTS


def main(do_train, do_test, nb_print_train, do_plot_train, do_save):
    experiment = EXPERIMENTS.no_quantum_learning.instantiate()
    experiment.launch(do_train, do_test, nb_print_train,
                      do_plot_train, do_save)
