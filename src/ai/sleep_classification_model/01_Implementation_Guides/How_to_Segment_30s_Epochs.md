# How to Segment 30s Epochs

## Objective
Segment continuous bio-signals (PPG/BP) into discrete 30-second windows for sleep stage classification.

## Steps
1. Load the continuous signal and state the sampling rate (`Fs`).
2. Calculate the number of samples per 30s epoch: `samples_per_epoch = 30 * Fs`.
3. Truncate the signal if its length is not a perfect multiple of `samples_per_epoch`.
4. Reshape or slice the continuous array into a matrix of shape `(num_epochs, samples_per_epoch)`.
5. Ensure hypnogram labels are perfectly aligned (1 sleep stage label per 30s epoch).
