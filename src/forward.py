import numpy as np


def forward_model(source, electrodes, dipole):
    potentials = []

    for electrode in electrodes:
        r_vector = electrode - source
        distance = np.linalg.norm(r_vector)
        dot_product = np.dot(dipole, r_vector)
        voltage = dot_product / (distance ** 3)
        potentials.append(voltage)

    return np.array(potentials)


def precompute_predictions(candidates, electrodes, dipole):
    predictions = []

    for candidate in candidates:
        predicted_voltage = forward_model(
            candidate,
            electrodes,
            dipole
        )
        predictions.append(predicted_voltage)

    return np.array(predictions)