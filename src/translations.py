import numpy as np
from itertools import groupby


def labels_to_states_simple(labels):
    return list(labels) # no transformation needed


def states_to_labels_simple(states):
    return states # no transformation needed


def labels_to_states_simple_v2(labels):
    converted_labels = []
    last_i_or_o = 'i'
    for l in labels:
        if l == 'i':
            last_i_or_o = 'i'
            converted_labels.append('i')
        elif l == 'o':
            last_i_or_o = 'o'
            converted_labels.append('o')
        elif last_i_or_o == 'i':
            converted_labels.append('M_io')
        elif last_i_or_o == 'o':
            converted_labels.append('M_oi')
        else:
            raise EnvironmentError("Some weird state.")  # just some error :D

    return converted_labels


def states_to_labels_simple_v2(states):
    return [s[0] for s in states]  # the first letter of the state is the expected outcome


def inverse(labels):
    inv = []
    for l in labels:
        if l == 'i':
            inv.append('o')
        elif l == 'o':
            inv.append('i')
        else:
            inv.append('M')

    return inv


def split(labels):
    return ["".join(g) for _, g in groupby(labels)]


def labels_to_states_complicated(labels):
    segments = split(labels)
    states = []
    prev_external = None
    for x in segments:
        if x[0] == 'i':
            prev_external = 'i'
            states += inside_to_states(len(x))
        elif x[0] == 'o':
            prev_external = 'o'
            states += outside_to_states(len(x))
        else:
            states += helix_to_states(len(x), 'io' if prev_external == 'i' else 'oi')

    return states


def inside_to_states(n):
    return ["i{}".format(s) for s in loop_to_states(n)]


def outside_to_states(n):
    return ["o{}".format(s) for s in loop_to_states(n)]


def loop_to_states(n):
    a = int(np.ceil(n / 2))
    b = n - a
    g = 0
    if b >= 10:
        g = a - 10 + b - 10
        a = 10
        b = 10

    return list(range(10, 10 + a)) + list('g' * g) + list(range(29 - b + 1, 29 + 1))


def helix_to_states(n, suffix):
    return ['M{}{}'.format(x, suffix) for x in get_helix_cap("a") + helix_core_statuses(n - 14) + get_helix_cap("b")]


def helix_core_statuses(n):
    return ["m"] + ["h{}".format(x + 1) for x in range(21 - n + 1, 21)]


def get_helix_cap(suffix):
    return ['c{}{}'.format(x + 1, suffix) for x in range(7)]


def states_to_labels_complicated(states):
    return states_to_labels_simple_v2(states)