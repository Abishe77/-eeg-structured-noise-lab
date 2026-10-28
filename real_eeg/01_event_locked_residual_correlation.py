import os
import mne
import numpy as np
import matplotlib.pyplot as plt

sample_path = r"D:\EEG work\EEG - MNE dataset\MNE-sample-data"
sample_meg_path = os.path.join(sample_path, "MEG", "sample")

raw_fname = os.path.join(
    sample_meg_path,
    "sample_audvis_raw.fif"
)

raw = mne.io.read_raw_fif(raw_fname, preload=False)

events = mne.find_events(raw)
event_dict = {
    "auditory/left" :1,
    "auditory/right" :2,
    "visual/left" : 3,
    "visual/right" :4,
    "smiley" :5,
    "buttonpress" :32
}

eeg_picks = mne.pick_types(
    raw.info , eeg= True , exclude = "bads"
)

epochs_1 = mne.Epochs(
    raw, events , event_id =1 , tmin = -0.2 ,tmax =0.5 ,picks = eeg_picks,baseline = None, preload = True
)

epoch_data_1 = epochs_1.get_data()
average_data = np.mean(epoch_data_1 , axis = 0) # mean across dimension one

#calculating sd around response from mean value
residuals = epoch_data_1 - average_data


epoch_times = epochs_1.times

values = residuals[ : , 0 , 200]
#for all 72 trials of EEG 001 at 200th time index we are going to find mean and std


'''At the first usable EEG electrode, around 133 ms after each left-ear tone, 
the measurements across the 72 trials deviate from their average response 
with an SD of about 15.9 µV this is measured value.'''

electrode1 = residuals[ : , 0 , 200]
electrode2 = residuals[ : , 1 , 200]

#EEG 001 and 002 residuals across 72 trials at 200th time index

#Calculating the cov between 001 and 002

deviation1 = electrode1 - np.mean(electrode1)
deviation2 = electrode2 - np.mean(electrode2)

product = deviation1 * deviation2
product_sum = np.sum(product)

denominator = len(electrode1) -1
cov_1_2 = product_sum / denominator

#The cov we got is 2.00625...

#Calculating std for both electrodes so that we can find r
std1 = np.std(electrode1 , ddof =1)
std2 = np.std(electrode2 , ddof =1)

r = cov_1_2 / (std1 * std2)
'''
residual_at_200 = residuals[: , : , 200]

cov_matrix = np.cov(residual_at_200, rowvar=False)
#since our column contains variables we choose that by making rowvar = False
correlation_matrix = np.corrcoef(residual_at_200, rowvar=False)
upper_indices = np.triu_indices(59, k=1)
pair_correlations = correlation_matrix[upper_indices]
'''

mean_correlations = []
upper_indices = np.triu_indices(59 , k=1)

for time_index in range(residuals.shape[2]):
    residual_at_time = residuals[: ,:, time_index]
    correlation_matrix = np.corrcoef(residual_at_time , rowvar = False)
    pair_correlation = correlation_matrix[upper_indices]
    r = np.mean(pair_correlation)
    mean_correlations.append(r)
'''
plt.plot(epoch_times, mean_correlations)
plt.axvline(0)
plt.xlabel("Time relative to event (s)")
plt.ylabel("Mean pairwise residual correlation")
plt.show()
'''
mean_correlations = np.array(mean_correlations)
min_index = np.argmin(mean_correlations)
max_index = np.argmax(mean_correlations)

print("minimum:", mean_correlations[min_index])
print("minimum time:", epoch_times[min_index])

print("maximum:", mean_correlations[max_index])
print("maximum time:", epoch_times[max_index])

pre_mean = np.mean(mean_correlations[epoch_times < 0])
post_mean = np.mean(mean_correlations[epoch_times > 0])

print("pre:", pre_mean)
print("post:", post_mean)

print(np.sum(epoch_times < 0))
print(np.sum(epoch_times > 0))

post_200ms = np.mean(
    mean_correlations[(epoch_times > 0) & (epoch_times <= 0.2)]
)

print("pre std:", np.std(mean_correlations[epoch_times < 0], ddof=1))

print(
    "post std:",
    np.std(
        mean_correlations[(epoch_times > 0) & (epoch_times <= 0.2)],
        ddof=1
    )
)

