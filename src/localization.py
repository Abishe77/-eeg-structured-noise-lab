import numpy as np


def localize_with_rmse(
    measured_potentials,
    predicted_voltages,
    candidates
):
    errors = []

    for prediction in predicted_voltages:
        difference = measured_potentials - prediction
        mse = np.mean(difference ** 2)
        rmse = np.sqrt(mse)
        errors.append(rmse)

    best_index = np.argmin(errors)

    return candidates[best_index]


def localize_with_covariance(
    measured_potentials,
    predicted_voltages,
    candidates,
    inverse_covariance
):
    scores = []

    for prediction in predicted_voltages:
        difference = measured_potentials - prediction

        score = (
            difference.T
            @ inverse_covariance
            @ difference
        )

        scores.append(score)

    best_index = np.argmin(scores)

    return candidates[best_index]


def localization_error(true_source, estimated_source):
    difference = estimated_source - true_source

    return np.linalg.norm(difference)