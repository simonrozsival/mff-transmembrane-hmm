from .hmm import HiddenMarkovModelTrainer


def create_trainer():
    return HiddenMarkovModelTrainer(symbols=list('ACDEFGHIKLMNPQRSTVWY'))
