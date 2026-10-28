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
    create_structured_covariance,
    generate_structured_noise
)

from src.localization import (
    localize_with_rmse,
    localize_with_covariance,
    localization_error
)


def add_independent_noise(clean_potentials, sigma):

    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=len(clean_potentials)
    )

    return clean_potentials + noise


def add_structured_noise(
    clean_potentials,
    covariance
):

    noise = generate_structured_noise(
        covariance
    )

    return clean_potentials + noise



# ADD STRUCTURED NOISE TO CLEAN MEASUREMENT


def add_structured_noise(
    clean_potentials,
    covariance
):

    noise = generate_structured_noise(
        covariance
    )

    return clean_potentials + noise



# USER INPUT


radius = int(
    input("Enter brain radius: ")
)

number_of_electrodes = int(
    input("Enter number of electrodes: ")
)



# TRUE SOURCE


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



# TRUE DIPOLE ORIENTATION


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



# BUILD ELECTRODES


electrodes = create_electrodes(
    radius,
    number_of_electrodes
)


print(
    "Electrode array shape:",
    electrodes.shape
)



# CLEAN FORWARD MEASUREMENT


clean_potentials = forward_model(
    true_source,
    electrodes,
    true_dipole
)



# BUILD CANDIDATE SOURCE GRID


candidates = create_candidates(
    radius,
    spacing=1
)


print(
    "Number of candidate sources:",
    len(candidates)
)



# PRECOMPUTE FORWARD PREDICTIONS
#
# Candidate dipoles use the TRUE orientation for now.
# Only source LOCATION is unknown.


predicted_voltages = precompute_predictions(
    candidates,
    electrodes,
    true_dipole
)


print(
    "Prediction matrix shape:",
    predicted_voltages.shape
)



# EXPERIMENT FUNCTION


def run_experiment(
    seed,
    correlation_length
):

    
    # RANDOM SEED
   

    np.random.seed(seed)


   
    # EXPERIMENT SETTINGS
    

    number_of_trials = 10000

    noise_sigma = 0.1


    
    # TRUE STRUCTURED COVARIANCE
    #
    # Used only to generate synthetic noise.
   

    structured_covariance = create_structured_covariance(
        electrodes,
        sigma=noise_sigma,
        correlation_length=correlation_length
    )


    
    # CALIBRATION NOISE
    

    number_of_calibration_samples = 10000

    calibration_noise = []

    for i in range(number_of_calibration_samples):

        noise = generate_structured_noise(
            structured_covariance
        )

        calibration_noise.append(noise)


    calibration_noise = np.array(
        calibration_noise
    )


   
    # ESTIMATE COVARIANCE
    

    lw = LedoitWolf()

    lw.fit(
        calibration_noise
    )

    estimated_covariance = lw.covariance_

    shrinkage = lw.shrinkage_


   
    # INVERT ESTIMATED COVARIANCE


    inverse_estimated_covariance = np.linalg.inv(
        estimated_covariance
    )


    
    # RESULT STORAGE
    

    rmse_errors = []

    cov_errors = []


    
    # MONTE CARLO TEST
   

    for trial in range(number_of_trials):

      
        measured_potentials = add_structured_noise(
            clean_potentials,
            structured_covariance
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




    rmse_errors = np.array(
        rmse_errors
    )

    cov_errors = np.array(
        cov_errors
    )


   

    rmse_recovery = np.mean(
        rmse_errors == 0
    )

    cov_recovery = np.mean(
        cov_errors == 0
    )


   
    rmse_mean_error = np.mean(
        rmse_errors
    )

    cov_mean_error = np.mean(
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


   

    return {

        "seed": seed,

        "correlation_length":
            correlation_length,

        "shrinkage":
            shrinkage,

        "rmse_recovery":
            rmse_recovery,

        "cov_recovery":
            cov_recovery,

        "improvement":
            cov_recovery - rmse_recovery,

        "rmse_mean_error":
            rmse_mean_error,

        "cov_mean_error":
            cov_mean_error,

        "both_correct":
            both_correct,

        "rmse_only_correct":
            rmse_only_correct,

        "cov_only_correct":
            cov_only_correct,

        "both_wrong":
            both_wrong
    }





seeds = [
    42,
    123,
    456,
    789,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000
]


correlation_length = 8.0


for seed in seeds:

    result = run_experiment(
        seed,
        correlation_length
    )

    print(
        "\n=============================="
    )

    print(
        "SEED:",
        result["seed"]
    )

    print(
        "Correlation length:",
        result["correlation_length"]
    )

    print(
        "Ledoit-Wolf shrinkage:",
        result["shrinkage"]
    )

    print(
        "RMSE recovery:",
        result["rmse_recovery"]
    )

    print(
        "Covariance-aware recovery:",
        result["cov_recovery"]
    )

    print(
        "Improvement:",
        result["improvement"]
    )

    print(
        "RMSE mean error:",
        result["rmse_mean_error"]
    )

    print(
        "Covariance mean error:",
        result["cov_mean_error"]
    )

    print(
        "Both correct:",
        result["both_correct"]
    )

    print(
        "RMSE only:",
        result["rmse_only_correct"]
    )

    print(
        "Covariance only:",
        result["cov_only_correct"]
    )

    print(
        "Both wrong:",
        result["both_wrong"]
    )