from src.parser import parse
# from src.translations import labels_to_states_simple as labels_to_states, states_to_labels_simple as states_to_labels  # naive model
# from src.translations import labels_to_states_simple_v2 as
# labels_to_states, states_to_labels_simple_v2 as states_to_labels  #
# naive model v2
from src.translations import labels_to_states_complicated as labels_to_states, states_to_labels_complicated as states_to_labels  # complex model
from src.training import train
import pickle


with open("set160.labels.txt") as file:
    sequences, labels = parse(file)
    train(sequences, labels, labels_to_states, states_to_labels)

    # # save the model to a file
    # name = "complex_model"
    # model_params = (trained_model._symbols, trained_model._states, trained_model._transitions, trained_model._outputs, trained_model._priors)
    # print(model_params)
    # pickle.dump(model_params, open(name, 'wb'))
