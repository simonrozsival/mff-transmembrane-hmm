import numpy as np

def parse(data_source):
    sequences =  np.array([])
    labels = np.array([])

    for line in data_source:
        _, sequence, label = line.split()
        sequences = np.append(sequences, [sequence])
        labels = np.append(labels, [label])

    return sequences, labels
