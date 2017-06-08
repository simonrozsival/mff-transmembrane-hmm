import numpy as np
from .translations import inverse, split


def calculate_error(labels, prediction):
    diff = [1 if l != p else 0 for l, p in zip(labels, prediction)]
    return np.mean(diff)


def eval(input, labels, prediction):
    error = calculate_error(labels, prediction)
    inv_error = calculate_error(labels, inverse(prediction))

    print("input:      ", input)
    print("labels:     ", "".join(labels))
    if inv_error < error:
        print("inv pred:   ", "".join(inverse(prediction)))
        print("inv error:  ", inv_error * 100, "%")
    else:
        print("prediction: ", "".join(prediction))
        print("error:      ", error * 100, "%")
    print()

    return error


def count_total_helixes(labels):
    labels_parts = split(labels)
    helixes = 0
    for part in labels_parts:
        if part[0] == 'M':
            helixes += 1

    return helixes


def under_predicted_helixes(labels, prediction):
    labels_parts = split(labels)
    over_predictions = 0
    for part in labels_parts:
        if part[0] == 'M':
            some_overlap = False
            for l in prediction[:len(part)]:
                if l == 'M':  # there is some overlap
                    some_overlap = True
                    break
            if some_overlap == False:
                over_predictions += 1
        prediction = prediction[len(part):]

    return over_predictions


def over_predicted_helixes(labels, prediction):
    return under_predicted_helixes(prediction, labels)


def shifted_helixes(labels, prediction):
    return 0


def falsely_merged_helixes(labels, prediction):
    return 0


def falsely_split_helixes(labels, prediction):
    return 0
