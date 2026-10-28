
import numpy as np
import mne




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


print("REAL EEG RESIDUAL ANALYSIS")


print(
    f"Residuals: {n_trials} trials × "
    f"{n_electrodes} electrodes × "
    f"{n_times} time points"
)



# MULTI-SEED COVARIANCE REPRODUCIBILITY


n_splits = 20

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

upper = np.triu_indices(
    n_electrodes,
    k=1
)

all_real = np.zeros(
    (len(seeds), n_times)
)

all_null = np.zeros(
    (len(seeds), n_times)
)

all_separation = np.zeros(
    (len(seeds), n_times)
)


for seed_index, seed in enumerate(seeds):

    rng = np.random.default_rng(seed)

   
    # REAL DATA
    

    stability_real = np.zeros(
        (n_splits, n_times)
    )

    for split in range(n_splits):

        indices = rng.permutation(
            n_trials
        )

        group_A = indices[:36]
        group_B = indices[36:]

        for t in range(n_times):

            A = residuals[
                group_A, :, t
            ]

            B = residuals[
                group_B, :, t
            ]

            cov_A = np.cov(
                A,
                rowvar=False
            )

            cov_B = np.cov(
                B,
                rowvar=False
            )

            r = np.corrcoef(
                cov_A[upper],
                cov_B[upper]
            )[0, 1]

            stability_real[
                split, t
            ] = r

    mean_real = np.mean(
        stability_real,
        axis=0
    )


   
    # TRIAL-SHUFFLED NULL
   

    null_residuals = residuals.copy()

    for electrode in range(
        n_electrodes
    ):

        shuffled_trials = rng.permutation(
            n_trials
        )

        null_residuals[
            :, electrode, :
        ] = residuals[
            shuffled_trials,
            electrode,
            :
        ]


    stability_null = np.zeros(
        (n_splits, n_times)
    )

    for split in range(n_splits):

        indices = rng.permutation(
            n_trials
        )

        group_A = indices[:36]
        group_B = indices[36:]

        for t in range(n_times):

            A = null_residuals[
                group_A, :, t
            ]

            B = null_residuals[
                group_B, :, t
            ]

            cov_A = np.cov(
                A,
                rowvar=False
            )

            cov_B = np.cov(
                B,
                rowvar=False
            )

            r = np.corrcoef(
                cov_A[upper],
                cov_B[upper]
            )[0, 1]

            stability_null[
                split, t
            ] = r

    mean_null = np.mean(
        stability_null,
        axis=0
    )

    separation = (
        mean_real - mean_null
    )

    all_real[
        seed_index
    ] = mean_real

    all_null[
        seed_index
    ] = mean_null

    all_separation[
        seed_index
    ] = separation



# ROBUSTNESS SUMMARY


mean_real_across_seeds = np.mean(
    all_real,
    axis=0
)

std_real_across_seeds = np.std(
    all_real,
    axis=0
)

mean_null_across_seeds = np.mean(
    all_null,
    axis=0
)

std_null_across_seeds = np.std(
    all_null,
    axis=0
)

mean_separation_across_seeds = np.mean(
    all_separation,
    axis=0
)

std_separation_across_seeds = np.std(
    all_separation,
    axis=0
)



print("COVARIANCE REPRODUCIBILITY")


selected_times = [
    0,
    120,
    200,
    300,
    420
]

for t in selected_times:

    print(
        f"{epoch_times[t]:+.3f} s | "
        f"Real: "
        f"{mean_real_across_seeds[t]:.4f} "
        f"+/- {std_real_across_seeds[t]:.4f} | "
        f"Null: "
        f"{mean_null_across_seeds[t]:.4f} "
        f"+/- {std_null_across_seeds[t]:.4f} | "
        f"Delta: "
        f"{mean_separation_across_seeds[t]:.4f}"
    )








