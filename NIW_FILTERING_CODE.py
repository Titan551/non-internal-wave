# NIW_FILTERING_CODE

#%%
import numpy as np
from scipy.signal import filtfilt

#%%
# Placeholder for the NIW filter function (assumes the same one translated earlier)
def NIW_filter_pl64t(x, cutoff=33):
    """
    Applies a low-pass filter to the time series x using the PL64 filter
    described in Rosenfeld (1983). Adapted for non-internal-wave estimates
    from coral reef temperature time series by Alex S.J. Wyatt et al. (2017).

    Parameters:
    x : array-like
        Input time series (1D or 2D array).
    cutoff : float, optional
        Cutoff frequency in samples. Default is 33 (e.g., representing 33 hours).

    Returns:
    xf : array-like
        The filtered time series with the same shape as x.
    """

    # Ensure x is a 2D array for consistent processing
    x = np.atleast_2d(x)
    npts, ncol = x.shape

    # If the input is a row vector, rotate to column vector
    rotated = False
    if npts == 1 and ncol > 1:
        x = x.T
        npts, ncol = x.shape
        rotated = True
    
    xf = np.copy(x)

    fq = 1.0 / cutoff
    nw = int(np.floor(2 * cutoff))
    nw2 = 2 * nw

    # Generate filter weights
    j = np.arange(1, nw + 1)
    t = np.pi * j
    den = (fq ** 2) * (t ** 3)
    wts = (2 * np.sin(2 * fq * t) - np.sin(fq * t) - np.sin(3 * fq * t)) / den

    # Symmetric filter weights and normalize to sum to one
    wts = np.concatenate([wts[::-1], [2 * fq], wts])
    wts /= np.sum(wts)

    # Cosine tapering
    cs = np.cos(t / nw2)
    jm = np.arange(nw, 0, -1)

    # Filter each column
    for ic in range(ncol):
        # Find finite (good) points
        jgd = np.isfinite(x[:, ic])
        ngd = np.sum(jgd)

        if ngd > nw2:
            # Detrend the time series
            jc = np.arange(np.where(jgd)[0][0], np.where(jgd)[0][-1] + 1)
            xdt = x[jc, ic]
            npts = len(jc)
            trnd = xdt[0] + (jc - jc[0]) * (xdt[-1] - xdt[0]) / (npts - 1)
            xdt = xdt - trnd

            # Fold and taper the time series on both ends
            y = np.concatenate([cs[jm - 1] * xdt[jm - 1], xdt, cs * xdt[::-1]])

            # Apply the filter
            yf = filtfilt(wts, 1.0, y)

            # Strip off extra points and add the trend back
            xf[jc, ic] = yf[nw2:npts + nw2] + trnd

        else:
            print('Warning: time series is too short')

    if rotated:
        xf = xf.T

    return xf

# Function to calculate the root mean square while ignoring NaN values
def nanrms(data):
    return np.sqrt(np.nanmean(data**2))

# Function to calculate max ignoring NaNs
def nanmax(data):
    return np.nanmax(data)

# Main NIW Processor
def NIW_processor(Temp, Time, x, sample_interval):
    """
    Parameters:
    Temp : array-like
        Bottom-mounted temperature time series sampled at e.g., 2-min intervals.
    Time : array-like
        Sampling time.
    x : float
        Local inertial frequency.
    sample_interval : int
        Samples per hour, e.g., 30 for 2-min data.

    Returns:
    NIW : array-like
        Final non-internal-wave (NIW) time series.
    """

    dtind = int(np.floor((x * sample_interval) / 2))  # Time interval index
    nsamps = int(sample_interval * 24 / 2)  # Number of samples in a day (divided by two for window)

    # Apply the low-pass filter
    LP = NIW_filter_pl64t(Temp, x * sample_interval)  # Low pass temperatures

    # High-pass filter (subtract low-pass from original)
    HP = Temp - LP

    # Calculate rms of the high-pass (HP) component
    rmsHP = np.empty_like(LP)
    tmax = np.empty_like(LP)

    for j in range(len(LP)):
        if j < dtind:
            rmsHP[j] = nanrms(HP[:j + dtind + 1])
        elif j > (len(LP) - dtind):
            rmsHP[j] = nanrms(HP[j - dtind:])
        else:
            rmsHP[j] = nanrms(HP[j - dtind:j + dtind + 1])

        # Calculate daily max Temp as a limit
        if j < nsamps:
            tmax[j] = nanmax(Temp[:j + nsamps + 1])
        elif j > (len(LP) - nsamps):
            tmax[j] = nanmax(Temp[j - nsamps:])
        else:
            tmax[j] = nanmax(Temp[j - nsamps:j + nsamps + 1])

    # Limit the NIW estimate to less than the daily max Temp
    NIWdummy = LP + rmsHP
    count = 0

    for j in range(len(NIWdummy)):
        if NIWdummy[j] > tmax[j]:
            NIWdummy[j] = tmax[j]
            count += 1

    print(f'Total of {count} points in NIW signal above running max set to running max ({int((count / len(NIWdummy)) * 100)}%)')

    # Final NIW time series
    NIW = NIW_filter_pl64t(NIWdummy, x * sample_interval)

    return NIW

#%%
# Example usage:
# Temp = ... # define the temperature data
# Time = ... # define the time data
# x = ...    # define the local inertial frequency
# sample_interval = 30 # e.g., for 2-min interval data

NIW_result = NIW_processor(Temp, Time, x, sample_interval)
