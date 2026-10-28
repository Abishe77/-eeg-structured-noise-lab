import numpy as np
import mne

from src.geometry import (
    create_electrodes,
    create_dipole,
    create_candidates
)

from src.forward import (
    forward_model,
    precompute_predictions
)

from src.localization import (
    localize_with_rmse,
    localize_with_covariance
)


raw_fname = (
    r"D:\EEG work\EEG - MNE dataset"
    r"\MNE-sample-data\MEG\sample"
    r"\sample_audvis_raw.fif"
)

raw = mne.io.read_raw_fif(
    raw_fname,
    preload=False
)

events = mne.find_events(raw)

eeg_picks = mne.pick_types(
    raw.info,
    eeg=True,
    exclude="bads"
)

epochs_1 = mne.Epochs(
    raw,
    events,
    event_id=1,
    tmin=-0.2,
    tmax=0.5,
    picks=eeg_picks,
    baseline=None,
    preload=True
)

epoch_data_1 = epochs_1.get_data()
epoch_times = epochs_1.times

average_data = np.mean(
    epoch_data_1,
    axis=0
)

residuals = epoch_data_1 - average_data

n_trials, n_electrodes, n_times = residuals.shape

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

print("REAL EEG RESIDUAL ANALYSIS")

print(
    f"Residuals: {n_trials} trials x "
    f"{n_electrodes} electrodes x "
    f"{n_times} time points"
)

eeg_channels = [
    ch for ch in raw.ch_names
    if raw.get_channel_types(
        picks=[ch]
    )[0] == "eeg"
    and ch not in raw.info["bads"]
]

electrode_positions = []

for ch in eeg_channels:
    idx = raw.ch_names.index(ch)

    electrode_positions.append(
        raw.info["chs"][idx]["loc"][:3]
    )

electrode_positions = np.array(
    electrode_positions
)

selected_indices = [0]

while len(selected_indices) < 8:
    selected_positions = (
        electrode_positions[
            selected_indices
        ]
    )

    distances = np.min(
        np.linalg.norm(
            electrode_positions[:, None, :]
            - selected_positions[None, :, :],
            axis=2
        ),
        axis=1
    )

    distances[selected_indices] = -1

    selected_indices.append(
        np.argmax(distances)
    )

selected_channels = [
    eeg_channels[i]
    for i in selected_indices
]

selected_electrode_indices = [
    eeg_picks.tolist().index(
        raw.ch_names.index(ch)
    )
    for ch in selected_channels
]

target_time_index = 200

selected_residuals = residuals[
    :,
    selected_electrode_indices,
    target_time_index
]

real_covariance = np.cov(
    selected_residuals,
    rowvar=False
)

std_devs = np.sqrt(
    np.diag(real_covariance)
)

real_correlation = (
    real_covariance
    / np.outer(
        std_devs,
        std_devs
    )
)

simulator_sigma = 0.1

real_eeg_sim_covariance = (
    simulator_sigma ** 2
    * real_correlation
)

eigenvalues = np.linalg.eigvalsh(
    real_eeg_sim_covariance
)

print("REAL EEG -> SIMULATOR COVARIANCE")

print(
    f"Source time: "
    f"{epoch_times[target_time_index]:+.3f} s"
)

print(
    "Selected electrodes:",
    ", ".join(selected_channels)
)

print(
    f"Correlation range: "
    f"{real_correlation.min():.3f} "
    f"to "
    f"{real_correlation.max():.3f}"
)

print(
    f"Minimum covariance eigenvalue: "
    f"{eigenvalues.min():.6f}"
)

sphere_radius = 3.0

true_source = np.array([
    1.0,
    2.0,
    1.0
])

candidate_spacing = 1.0

n_test = 10000

n_sim_electrodes = 8

electrodes = create_electrodes(
    sphere_radius,
    n_sim_electrodes
)

true_dipole = create_dipole(
    90,
    0
)

candidates = create_candidates(
    sphere_radius,
    candidate_spacing
)

prediction_matrix = precompute_predictions(
    candidates,
    electrodes,
    true_dipole
)

true_measurement_clean = forward_model(
    true_source,
    electrodes,
    true_dipole
)

inverse_covariance = np.linalg.inv(
    real_eeg_sim_covariance
)

results = []

print("FINAL BRIDGE EXPERIMENT")

for seed in seeds:

    rng = np.random.default_rng(
        seed
    )

    rmse_correct = 0
    covariance_correct = 0

    rmse_errors = []
    covariance_errors = []

    for _ in range(n_test):

        noise = (
            rng.multivariate_normal(
                mean=np.zeros(
                    n_sim_electrodes
                ),
                cov=real_eeg_sim_covariance
            )
        )

        measured = (
            true_measurement_clean
            + noise
        )

        rmse_location = localize_with_rmse(
            measured,
            prediction_matrix,
            candidates
        )

        covariance_location = (
            localize_with_covariance(
                measured,
                prediction_matrix,
                candidates,
                inverse_covariance
            )
        )

        rmse_error = np.linalg.norm(
            rmse_location
            - true_source
        )

        covariance_error = np.linalg.norm(
            covariance_location
            - true_source
        )

        rmse_errors.append(
            rmse_error
        )

        covariance_errors.append(
            covariance_error
        )

        if rmse_error == 0:
            rmse_correct += 1

        if covariance_error == 0:
            covariance_correct += 1

    rmse_accuracy = (
        rmse_correct / n_test
    )

    covariance_accuracy = (
        covariance_correct / n_test
    )

    rmse_mean_error = np.mean(
        rmse_errors
    )

    covariance_mean_error = np.mean(
        covariance_errors
    )

    results.append([
        rmse_accuracy,
        covariance_accuracy,
        rmse_mean_error,
        covariance_mean_error
    ])

results = np.array(results)

rmse_accuracy = results[:, 0]
covariance_accuracy = results[:, 1]

rmse_error = results[:, 2]
covariance_error = results[:, 3]

accuracy_difference = (
    covariance_accuracy
    - rmse_accuracy
)

error_difference = (
    rmse_error
    - covariance_error
)

print("FINAL RESULT")

print(
    f"RMSE accuracy: "
    f"{rmse_accuracy.mean() * 100:.2f}% "
    f"+/- "
    f"{rmse_accuracy.std() * 100:.2f}%"
)

print(
    f"Covariance-aware accuracy: "
    f"{covariance_accuracy.mean() * 100:.2f}% "
    f"+/- "
    f"{covariance_accuracy.std() * 100:.2f}%"
)

print(
    f"Accuracy improvement: "
    f"{accuracy_difference.mean() * 100:.2f} "
    f"percentage points"
)

print()

print(
    f"RMSE mean localization error: "
    f"{rmse_error.mean():.6f}"
)

print(
    f"Covariance-aware mean localization error: "
    f"{covariance_error.mean():.6f}"
)

print(
    f"Mean localization error reduction: "
    f"{error_difference.mean():.6f}"
)

print("RESEARCH EXPERIMENT COMPLETE")


