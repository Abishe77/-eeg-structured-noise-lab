import numpy as np
from sklearn.covariance import LedoitWolf

from src.geometry import (
    create_electrodes,
    create_dipole,
    create_candidates
)

from src.forward import (
    forward_model,
    precompute_predictions
)

from src.noise import (
    generate_independent_noise
)

from src.localization import (
    localize_with_rmse,
    localize_with_covariance,
    localization_error
)




def summarize_results(errors):

    print("Mean error:", np.mean(errors))
    print("Median error:", np.median(errors))
    print("SD:", np.std(errors))
    print("Minimum error:", np.min(errors))
    print("Maximum error:", np.max(errors))
    print(
        "Exact recovery rate:",
        np.mean(errors == 0)
    )


radius = int(
    input("Enter brain radius: ")
)

number_of_electrodes = int(
    input("Enter number of electrodes: ")
)




while True:

    source_x = int(
        input("Source x: ")
    )

    source_y = int(
        input("Source y: ")
    )

    source_z = int(
        input("Source z: ")
    )

    if (
        source_x ** 2
        + source_y ** 2
        + source_z ** 2
        < radius ** 2
    ):
        break

    print("Source must be inside the sphere.")


true_source = np.array([
    source_x,
    source_y,
    source_z
])


print("\nTrue source:", true_source)



phi = int(
    input("Dipole phi (look up/down): ")
)

theta = int(
    input("Dipole theta (rotation): ")
)


true_dipole = create_dipole(
    phi,
    theta
)


print("Dipole vector:", true_dipole)




electrodes = create_electrodes(
    radius,
    number_of_electrodes
)


print(
    "Electrode array shape:",
    electrodes.shape
)



clean_potentials = forward_model(
    true_source,
    electrodes,
    true_dipole
)




candidates = create_candidates(
    radius,
    spacing=1
)


print(
    "Number of candidate sources:",
    len(candidates)
)


predicted_voltages = precompute_predictions(
    candidates,
    electrodes,
    true_dipole
)


print(
    "Prediction matrix shape:",
    predicted_voltages.shape
)




np.random.seed(42)

number_of_calibration_samples = 10000

number_of_trials = 10000

noise_sigma = 0.1




calibration_noise = []

for i in range(number_of_calibration_samples):

    noise = generate_independent_noise(
        number_of_electrodes,
        noise_sigma
    )

    calibration_noise.append(noise)


calibration_noise = np.array(
    calibration_noise
)


print(
    "\nCalibration noise shape:",
    calibration_noise.shape
)




lw = LedoitWolf()

lw.fit(
    calibration_noise
)


estimated_covariance = lw.covariance_

shrinkage = lw.shrinkage_


print(
    "Estimated covariance shape:",
    estimated_covariance.shape
)

print(
    "Ledoit-Wolf shrinkage:",
    shrinkage
)




inverse_estimated_covariance = np.linalg.inv(
    estimated_covariance
)




rmse_errors = []

cov_errors = []

rmse_results = {}

cov_results = {}



for trial in range(number_of_trials):


    noise = generate_independent_noise(
        number_of_electrodes,
        noise_sigma
    )

    measured_potentials = (
        clean_potentials
        + noise
    )




    rmse_estimated_source = localize_with_rmse(
        measured_potentials,
        predicted_voltages,
        candidates
    )

    rmse_error = localization_error(
        true_source,
        rmse_estimated_source
    )

    rmse_errors.append(
        rmse_error
    )


    rmse_location = tuple(
        rmse_estimated_source
    )

    if rmse_location in rmse_results:

        rmse_results[rmse_location] += 1

    else:

        rmse_results[rmse_location] = 1




    cov_estimated_source = localize_with_covariance(
        measured_potentials,
        predicted_voltages,
        candidates,
        inverse_estimated_covariance
    )

    cov_error = localization_error(
        true_source,
        cov_estimated_source
    )

    cov_errors.append(
        cov_error
    )


    cov_location = tuple(
        cov_estimated_source
    )

    if cov_location in cov_results:

        cov_results[cov_location] += 1

    else:

        cov_results[cov_location] = 1




rmse_errors = np.array(
    rmse_errors
)

cov_errors = np.array(
    cov_errors
)




both_correct = np.sum(
    (rmse_errors == 0)
    &
    (cov_errors == 0)
)

rmse_only_correct = np.sum(
    (rmse_errors == 0)
    &
    (cov_errors != 0)
)

cov_only_correct = np.sum(
    (rmse_errors != 0)
    &
    (cov_errors == 0)
)

both_wrong = np.sum(
    (rmse_errors != 0)
    &
    (cov_errors != 0)
)




covariance_wins = np.sum(
    cov_errors < rmse_errors
)

rmse_wins = np.sum(
    rmse_errors < cov_errors
)

ties = np.sum(
    rmse_errors == cov_errors
)





print("INDEPENDENT NOISE + RMSE")


print("Estimated source frequencies:")

for location, count in sorted(
    rmse_results.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        location,
        ":",
        count
    )


summarize_results(
    rmse_errors
)






print(
    "INDEPENDENT NOISE + ESTIMATED COVARIANCE"
)




print(
    "Estimated source frequencies:"
)


for location, count in sorted(
    cov_results.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        location,
        ":",
        count
    )


summarize_results(
    cov_errors
)

print("PAIRED COMPARISON")



print(
    "Both correct:",
    both_correct
)

print(
    "RMSE only correct:",
    rmse_only_correct
)

print(
    "Covariance-aware only correct:",
    cov_only_correct
)

print(
    "Both wrong:",
    both_wrong
)





print("LOCALIZATION ERROR COMPARISON")



print(
    "Covariance-aware lower error:",
    covariance_wins
)

print(
    "RMSE lower error:",
    rmse_wins
)

print(
    "Equal error:",
    ties
)
