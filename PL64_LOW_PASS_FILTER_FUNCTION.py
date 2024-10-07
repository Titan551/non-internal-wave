# PL64_LOW_PASS_FILTER_FUNCTION

#%%
import numpy as np
from scipy.signal import filtfilt

#%%
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
