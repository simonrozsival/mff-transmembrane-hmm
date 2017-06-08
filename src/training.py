import numpy as np
from sklearn.model_selection import KFold
from .eval import eval, count_total_helixes, over_predicted_helixes, under_predicted_helixes, shifted_helixes, falsely_merged_helixes, falsely_split_helixes
from .model import create_trainer


def train(sequences, labels, labels_to_states, states_to_labels):
    results = []
    stats = []
    translated_labels = np.array([labels_to_states(l) for l in labels])

    # 10-fold
    kf = KFold(n_splits=10, shuffle=True)

    for train, test in kf.split(sequences):
        # train
        X_train = sequences[train]
        y_train = translated_labels[train]
        data = [list(zip(x, y)) for x, y in zip(X_train, y_train)]
        trainer = create_trainer()
        trained_model = trainer.train_supervised(data)

        # test
        X_test = sequences[test]
        y_test = translated_labels[test]
        errors = []

        total_helixes = 0
        over_predicted = 0
        under_predicted = 0
        shifted = 0
        falsely_merged = 0
        falsely_split = 0

        for x, y in zip(X_test, y_test):
            prediction = list(trained_model.best_path(x))
            error = eval(x, states_to_labels(y), states_to_labels(prediction))
            t = count_total_helixes(states_to_labels(y))
            total_helixes += t
            o = over_predicted_helixes(
                states_to_labels(y), states_to_labels(prediction))
            over_predicted += o
            u = under_predicted_helixes(
                states_to_labels(y), states_to_labels(prediction))
            under_predicted += u
            s = shifted_helixes(states_to_labels(
                y), states_to_labels(prediction))
            shifted += s
            fm = falsely_merged_helixes(
                states_to_labels(y), states_to_labels(prediction))
            falsely_merged += fm
            fs = falsely_split_helixes(
                states_to_labels(y), states_to_labels(prediction))
            falsely_split += fs

            print(">    total helixes:   {}".format(t))
            print(">    over predicted:  {} ({} %)".format(o, o / t * 100))
            print(">    under predicted: {} ({} %)".format(u, u / t * 100))
            print()
            print("***")
            print()

            # print(">    shifted:         {} ({} %)".format(s, s/ t* 100))
            # print(">    falsely merged:  {} ({} %)".format(fm, fm/ t* 100))
            # print(">    falsely split:   {} ({} %)".format(fs, fs/ t* 100))

            errors.append(error)

        stats.append([total_helixes, over_predicted, under_predicted,
                      shifted, falsely_merged, falsely_split])
        error = np.mean(errors)
        success = 1 - error
        results.append(success)

    print("**********************")
    print(" Results:")
    print("**********************")
    for i in range(len(results)):
        total_helixes, over_predicted, under_predicted, shifted, falsely_merged, falsely_split = stats[
            i]
        print("#{} average success:  {} % (letter by letter)".format(
            i + 1, results[i] * 100))
        print(">    total helixes:   {}".format(total_helixes))
        print(">    over predicted:  {} ({} %)".format(
            over_predicted, over_predicted / total_helixes * 100))
        print(">    under predicted: {} ({} %)".format(
            under_predicted, under_predicted / total_helixes * 100))
        # print(">    shifted:         {} ({} %)".format(shifted, shifted / total_helixes * 100))
        # print(">    falsely merged:  {} ({} %)".format(falsely_merged, falsely_merged/ total_helixes * 100))
        # print(">    falsely split:   {} ({} %)".format(falsely_split, falsely_split / total_helixes * 100))

    # average everything
    total_helixes, over_predicted, under_predicted, shifted, falsely_merged, falsely_split = np.sum(
        stats, axis=0)

    print()
    print('**********************')
    print("Final score: {} %".format(np.mean(results) * 100))
    print(">    total helixes:   {}".format(total_helixes))
    print(">    over predicted:  {} ({} %)".format(
        over_predicted, over_predicted / total_helixes * 100))
    print(">    under predicted: {} ({} %)".format(
        under_predicted, under_predicted / total_helixes * 100))
    # print(">    shifted:         {} ({} %)".format(shifted, shifted / total_helixes * 100))
    # print(">    falsely merged:  {} ({} %)".format(falsely_merged, falsely_merged / total_helixes * 100))
    # print(">    falsely split:   {} ({} %)".format(falsely_split, falsely_split / total_helixes * 100))
