import numpy as np


def recall_at_k(model, batch_test, k):
    """
    Computes the recall at k metric for a model.
    :param model: The model to evaluate.
    :param k: The number of items to consider.
    :return: The recall at k metric.
    """
    # hits = 0
    # total = 0
    rec = 0
    for i, user in enumerate(batch_test):
        actual_results = model.lst[user]
        train_results = model.train_lst[user]
        actual_results = np.setdiff1d(actual_results, train_results)
        if len(actual_results) == 0:
            continue
        algo_results = model.predict(user, len(actual_results))
        algo_results = np.setdiff1d(algo_results, train_results)
        numerator = np.intersect1d(algo_results, actual_results)
        rec += len(numerator) / len(actual_results)
        # if len(numerator) > 0:
        #     print(f"User: {i}")
        #     print(f"Intersection: {numerator}")
        # hits += len(numerator)
        # total += len(actual_results)

    return rec / len(batch_test)
