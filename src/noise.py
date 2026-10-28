import numpy as np


def generate_independent_noise(number_of_electrodes, sigma):
    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=number_of_electrodes
    )

    return noise


def create_structured_covariance(
    electrodes,
    sigma=0.1,
    correlation_length=2.0
):
    count_electrodes = len(electrodes)

    covariance = np.zeros(
        (count_electrodes, count_electrodes)
    )

    for i in range(count_electrodes):
        for j in range(count_electrodes):
            distance = np.sqrt(
                (electrodes[i][0] - electrodes[j][0]) ** 2
                + (electrodes[i][1] - electrodes[j][1]) ** 2
                + (electrodes[i][2] - electrodes[j][2]) ** 2
            )

            covariance_value = (
                sigma ** 2
                * np.exp(
                    -distance / correlation_length
                )
            )

            covariance[i][j] = covariance_value

    return covariance


def generate_structured_noise(covariance):
    mean = np.zeros(
        covariance.shape[0]
    )

    noise = np.random.multivariate_normal(
        mean=mean,
        cov=covariance
    )

    return noise