import numpy as np


def recall_at_k(model, batch_test, k):
    """
    Computes the recall at k metric for a model.
    :param model: The model to evaluate.
    :param k: The number of items to consider.
    :return: The recall at k metric.
    """
    hits = 0
    total = 0
    for i, user in enumerate(batch_test):
        actual_results = model.lst[user]

        algo_results = model.predict(user, k)
        numerator = np.intersect1d(algo_results, actual_results)

        hits += len(numerator)
        total += len(actual_results)

    return hits / total
