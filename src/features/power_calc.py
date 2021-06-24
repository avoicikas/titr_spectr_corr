# calculate power - save power (individual)
# simple power, SNR and foof and bycyle
# calculate profiles - save profiles (all in one csv)
# calculate correlations
import mne
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.records import fromarrays
from scipy.io import savemat
from scipy import signal
import re

def import_eeg(ifile):
    data = mne.io.read_epochs_eeglab(ifile, eog="auto")
    data.load_data()
    data.set_montage("standard_1020")
    data.set_eeg_reference(ref_channels="average")
    return data

def calculate_spectrum(data):
    from scipy.signal import welch
    sf = data.info['sfreq']
    chan = data.ch_names
    values = epochs.pick_channels(['Fz']).data *1e6
    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(values, sf, nperseg=win)  # Works with single or multi-channel data
    print(freqs.shape, psd.shape)  # psd has shape (n_channels, n_frequencies)
    # Plot
    plt.plot(freqs, psd[0, :], 'k', lw=2)
    plt.fill_between(freqs, psd[1, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)');


def foof_():
    from fooof import FOOOFGroup
    from fooof.bands import Bands
    from fooof.analysis import get_band_peak_fg
    from fooof.plts.spectra import plot_spectrum
    # Initialize a FOOOFGroup object, with desired settings
    # Define the frequency range to fit
    freq_range = [1, 80]
    fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=.15, peak_threshold=2., max_n_peaks=1, verbose=False)
    # Fit the power spectrum model across all channels
    fg.fit(freqs, spectra, freq_range)
    # Check the overall results of the group fits
    fg.plot()
    # Define frequency bands of interest
    bands = Bands({'theta': [3, 7],
                   'alpha': [7, 14],
                   'gamma': [30, 60],
                   'beta': [15, 30]})
    # Extract alpha peaks
    alphas = get_band_peak_fg(fg, bands.beta)
    # Extract the power values from the detected peaks
    alpha_pw = alphas[:, 1]
    # Plot the topography of alpha power
    mne.viz.plot_topomap(alpha_pw, erp.info, cmap=cm.viridis, contours=0);
    # Plot the topographies across different frequency bands
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ind, (label, band_def) in enumerate(bands):
        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
        # Create a topomap for the current oscillation band
        mne.viz.plot_topomap(band_power, erp.info, cmap=cm.viridis, contours=0,
                             axes=axes[ind], show=False);
        # Set the plot title
        axes[ind].set_title(label + ' power', {'fontsize' : 20})
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    for ind, (label, band_def) in enumerate(bands):
        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
        # Extracted and plot the power spectrum model with the most band power
        fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)
        # Set some plot aesthetics & plot title
        axes[ind].yaxis.set_ticklabels([])
        axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})
    # Extract aperiodic exponent values
    exps = fg.get_params('aperiodic_params', 'exponent')
    # Plot the topography of aperiodic exponents
    plot_topomap(exps, erp.info, cmap=cm.viridis, contours=0)
    # Compare the power spectra between low and high exponent channels
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_spectrum(fg.freqs, fg.get_fooof(np.argmin(exps)).power_spectrum,
                  ax=ax, label='Low Exponent')
    plot_spectrum(fg.freqs, fg.get_fooof(np.argmax(exps)).power_spectrum,
                  ax=ax, label='High Exponent')

def get_power(data, tim1, tim2):
    sf = int(data.info['sfreq'])
    spectra, freqs = mne.time_frequency.psd_welch(data, fmin=1, fmax=80, tmin=tim1, tmax=tim2, n_overlap=0, n_fft=sf, n_per_seg=sf)
    return spectra, freqs

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()
    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )
    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )
    return psd / mean_noise


def plot_psd_snr():
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))
    psds_plot = 10 * np.log10(spectra)
    psds_mean = psds_plot.mean(axis=(0, 1))
    psds_std = psds_plot.std(axis=(0, 1))
    axes[0].plot(freqs, psds_mean, color='b')
    axes[0].fill_between(
        freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
        color='b', alpha=.2)
    axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')
    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]
    axes[1].plot(freqs[freq_range], snr_mean, color='r')
    axes[1].fill_between(
        freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
        color='r', alpha=.2)
    axes[1].set(
        title="SNR spectrum", xlabel='Frequency [Hz]',
        ylabel='SNR', ylim=[-2, 30], xlim=[fmin, fmax])


def save_power(spectra, freqs, erp, saveName, filename, saveDir):
    # csv: subjname, stimfreq, channel, freq, spectr
    subjname = re.findall(r'(.*)-',filename)[0]
    stimfreq = re.findall(r'-(\d+)',filename)[0]
    csvfile = saveDir.joinpath(saveName + '.csv')
    if not pathlib.Path(csvfile).exists():
        csvfile.write_text(f'subjname,stimfreq,ichan,ifreq,spectra'+'\n')
    with open(csvfile,'a') as f:
        for idxchan, ichan in enumerate(erp.ch_names):
            for idxfreq, ifreq in enumerate(freqs):
                f.write(f'{subjname},{stimfreq},{ichan},{ifreq},{spectra[idxchan, idxfreq]}'+'\n')


def calculate_powers():
    data_dir = "../../data/raw"
    failai = list(pathlib.Path(data_dir).glob("*.set"))
    saveDir = pathlib.Path("../../data/interim/powers")
    saveDir.mkdir(exist_ok=True)
    for ifile in failai:
        data = import_eeg(ifile)
        data= data.set_eeg_reference()
        erp = data.average()
        # get power total
        spectra, freqs = get_power(data,0,.5)
        spectra = spectra.mean(axis=0)
        save_power(spectra, freqs, erp, 'total_power', ifile.stem, saveDir)
        # total power snr
        snrs = snr_spectrum(spectra, noise_n_neighbor_freqs=6, noise_skip_neighbor_freqs=1)
        save_power(snrs, freqs, erp, 'total_power_snr', ifile.stem, saveDir)
        # get power total bl
        spectra_bl, freqs = get_power(data,-0.5,0)
        spectra_bl = spectra_bl.mean(axis=0)
        save_power(spectra_bl, freqs, erp, 'total_power_bl', ifile.stem, saveDir)
        # total ratio
        spectra = spectra/spectra_bl
        save_power(spectra, freqs, erp, 'total_power_ratio', ifile.stem, saveDir)
        # erp power
        spectra, freqs = get_power(erp,0,.5)
        save_power(spectra, freqs, erp, 'erp_power', ifile.stem, saveDir)
        # erp snr
        snrs = snr_spectrum(spectra, noise_n_neighbor_freqs=6, noise_skip_neighbor_freqs=1)
        save_power(snrs, freqs, erp, 'erp_snr', ifile.stem, saveDir)
        # erp baseline
        spectra_bl, freqs = get_power(erp,-0.5,0)
        save_power(spectra_bl, freqs, erp, 'erp_power_bl', ifile.stem, saveDir)
        # erp ratio
        spectra = spectra/spectra_bl
        save_power(spectra, freqs, erp, 'erp_power_bl', ifile.stem, saveDir)

def get_profile():
    pass

def igf():


if __name__ == "__main__":
    plot_psds()
