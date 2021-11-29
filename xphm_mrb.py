import numpy as np
import scipy as sp
import lal
import xphm
import gwpy
from gwpy.timeseries import TimeSeries

class data():
    """
    A class that gets information from strain data from LIGO/Virgo data releases.
    """
    def __init__(self, T, Fs, T_long, tgps, tukey_alpha = 0.2, detstrings = ['H1', 'L1', 'V1']):
        """
        Initializes the data object, which contains a time series, a frequency series, and a psd for each of the LIGO/Virgo detectors specified in detstrings.
        T: length of time series (s)
        Fs: sampling rate (Hz)
        T_long: length of longer time series used to compute the psd using Welch's method (s)
        tgps: the approximate gps time of the event
        tukey_alpha: the alpha parameter in the tukey window used in the fourier transform to get the frequency series
        detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
        """
        self.ndet = len(detstrings)
        self.detstrings = detstrings
        
        self.strain_td = np.zeros([self.ndet, T*Fs])
        self.strain_fd = np.zeros([self.ndet, T*Fs//2+1], dtype = complex)
        self.psd = np.zeros([self.ndet, T*Fs//2+1])
        self.ts = np.linspace(-T/2, T/2-1/Fs, T*Fs)
        
        for detstring, i in zip(detstrings, range(self.ndet)):
            data_long = TimeSeries.fetch_open_data(detstring, tgps-T_long/2, tgps+T_long/2).resample(Fs)
            data_short = TimeSeries.fetch_open_data(detstring, tgps-T/2, tgps+T/2).resample(Fs)

            strain_td_long = np.array(data_long)
            self.strain_td[i] = np.array(data_short)
            self.strain_fd[i] = 1/Fs*np.fft.rfft(self.strain_td[i]*sp.signal.tukey(len(self.strain_td[i]), alpha=tukey_alpha))

            self.fs, self.psd[i] = sp.signal.welch(strain_td_long, fs=Fs, nperseg = Fs*T)
        self.T = T
        self.Fs = Fs
        self.tgps = tgps
        
    def whitened_data(self):
        """
        Gets the whitened (unit covariance) time series for the data and the corresponding times since tgps.
        """
        whitened_strain = np.array([whiten_fd(self.strain_fd[i], self.psd[i], self.Fs) for i in range(self.ndet)])
        return self.ts, whitened_strain
    
class injected_data():
    """
    A class that creates fake strain data by injecting a false signal into noise from LIGO/Virgo data releases.
    """
    def __init__(self, T, Fs, T_long, tgps, pdict, uselal=False, tukey_alpha = 0.2, detstrings = ['H1', 'L1', 'V1']):
        """
        Initializes a injected_data object, which has the same properties as a regular data object: it contains a time series, a frequency series, and a psd for each of the LIGO/Virgo detectors specified in detstrings.
        T: length of time series (s)
        Fs: sampling rate (Hz)
        T_long: length of longer time series used to compute the psd using Welch's method (s)
        tgps: a gps time where there should be only noise
        tukey_alpha: the alpha parameter in the tukey window used in the fourier transform to get the frequency series
        detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
        """
        
        self.ndet = len(detstrings)
        self.detstrings = detstrings
        
        self.strain_td = np.zeros([self.ndet, T*Fs])
        self.strain_fd = np.zeros([self.ndet, T*Fs//2+1], dtype = complex)
        self.psd = np.zeros([self.ndet, T*Fs//2+1])
        self.ts = np.linspace(-T/2, T/2-1/Fs, T*Fs)
        
        # create waveform to inject:
        waveform = model(T, Fs, pdict, uselal=uselal, detstrings=detstrings)
        
        for detstring, i in zip(detstrings, range(self.ndet)):
            data_long = TimeSeries.fetch_open_data(detstring, tgps-T_long/2, tgps+T_long/2).resample(Fs)
            data_short = TimeSeries.fetch_open_data(detstring, tgps-T/2, tgps+T/2).resample(Fs)

            strain_td_long = np.array(data_long)
            self.strain_fd[i] = 1/Fs*np.fft.rfft(data_short*sp.signal.tukey(len(data_short), alpha=tukey_alpha)) + waveform.strain_fd[i]
            self.strain_td[i] = Fs*np.fft.irfft(self.strain_fd[i])

            self.fs, self.psd[i] = sp.signal.welch(strain_td_long, fs=Fs, nperseg = Fs*T)
        self.T = T
        self.Fs = Fs
        self.tgps = tgps
        
    def whitened_data(self):
        """
        Gets the whitened (unit covariance) time series for the data and the corresponding times since tgps.
        """
        whitened_strain = np.array([whiten_fd(self.strain_fd[i], self.psd[i], self.Fs) for i in range(self.ndet)])
        return self.ts, whitened_strain
    
class model():
    """
    A class that uses xphm.py to build the IMRPhenomXPHM model.
    """
    def __init__(self, T, Fs, pdict, uselal = False, compute_modes = False, compute_td_L_frame_modes = False, compute_C_coefficients = False, detstrings = ['H1', 'L1', 'V1']):
        """
        Initializes the model object, which contains a model frequency series for each of the LIGO/Virgo detectors specified in detstrings.
        T: length of time series (s)
        Fs: sampling rate (Hz)
        pdict: parameter dictionary for the model evaluation
        uselal: boolean, if True: the model uses lal to get the plus/cross modes,
                         if False: the model uses lal to get the euler angles and L frame modes and builds the plus/cross modes from these
        compute_modes: boolean, if True: the model additionally computes an array of the mode components of the strain
        compute_td_L_frame_modes: boolean, if True: the model additionally computes an array of the time dependent L-frame modes k_lm
        compute_C_coefficients: boolean, if True: an array of the coefficients C_lm of the time dependent L-frame modes
        detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
        """
        f_min = pdict['f_min']
        f_max = pdict['f_max']
        fs_model = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1)
        fs_model_seq = lal.CreateREAL8Sequence(len(fs_model))
        fs_model_seq.data = fs_model
        
        self.ndet = len(detstrings)
        self.detstrings = detstrings
        model_nonzero = xphm.compute_strain(pdict, fs_model_seq, uselal=uselal, detstrings=detstrings)
        model = np.zeros([self.ndet,T*Fs//2+1], dtype=complex)
        model[:,int(f_min*T):int(f_max*T)+1] = model_nonzero
        
        self.strain_fd = model
        self.T = T
        self.Fs = Fs
        self.fmin = f_min
        self.fmax = f_max
        self.pdict = pdict
        
        if compute_modes:
            self.compute_modes()
    
    def compute_modes(self):
        """
        Computes an array of the mode components of the strain. Indices are (detector, mode, frequency). A sum over the mode indices produces the frequency domain strain in each detector.
        """
        f_min = self.pdict['f_min']
        f_max = self.pdict['f_max']
        T = self.T
        Fs = self.Fs
        fs_model = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1)
        fs_model_seq = lal.CreateREAL8Sequence(len(fs_model))
        fs_model_seq.data = fs_model
        
        modes_nonzero = xphm.compute_strain_components(self.pdict, fs_model_seq, detstrings=self.detstrings)
        modes = np.zeros([self.ndet,5,T*Fs//2+1], dtype=complex)
        modes[:, :, int(f_min*T):int(f_max*T)+1] = modes_nonzero
        self.modes = modes
        
    def compute_td_L_frame_modes(self):
        """
        Computes an array of the time dependent L-frame modes k_lm. Indices are (detector, mode, frequency). A sum of these times the C coefficients over the mode indices produces the frequency domain strain in each detector.
        """
        f_min = self.pdict['f_min']
        f_max = self.pdict['f_max']
        T = self.T
        Fs = self.Fs
        fs_model = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1)
        fs_model_seq = lal.CreateREAL8Sequence(len(fs_model))
        fs_model_seq.data = fs_model
        
        td_L_frame_modes_nonzero = xphm.compute_td_L_frame_modes(self.pdict, fs_model_seq, detstrings=self.detstrings)
        td_L_frame_modes = np.zeros([self.ndet,5,T*Fs//2+1], dtype=complex)
        td_L_frame_modes[:, :, int(f_min*T):int(f_max*T)+1] = td_L_frame_modes_nonzero
        self.td_L_frame_modes = td_L_frame_modes
        
    def compute_C_coefficients(self):
        """
        Computes an array of the coefficients C_lm of the time dependent L-frame modes. Indices are (detector, mode, frequency). A sum of these times the time-dependent L-frame modes over the mode indices produces the frequency domain strain in each detector.
        """
        f_min = self.pdict['f_min']
        f_max = self.pdict['f_max']
        T = self.T
        Fs = self.Fs
        fs_model = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1)
        fs_model_seq = lal.CreateREAL8Sequence(len(fs_model))
        fs_model_seq.data = fs_model
        
        Ccoeffs_nonzero = xphm.compute_C_prefactors(self.pdict, fs_model_seq, detstrings=self.detstrings)
        Ccoeffs = np.zeros([self.ndet,5,T*Fs//2+1], dtype=complex)
        Ccoeffs[:, :, int(f_min*T):int(f_max*T)+1] = Ccoeffs_nonzero
        self.Ccoeffs = Ccoeffs
    
    def whitened_model(self, data):
        """
        Gets the whitened (unit covariance) time series for the model and the corresponding times since tgps.
        data: data object
        """
        whitened_strain = np.array([whiten_fd(self.strain_fd[i], data.psd[i], self.Fs) for i in range(self.ndet)])
        return whitened_strain
        
def inn_prod(x, y, psd, T):
    """
    Computes the inner product between two frequency series.
    x, y: frequency series arrays
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    """
    return np.real(np.sum(x*np.conj(y)/(psd*T/4)))

def time_delay_overlap(data, model):
    """
    Computes the overlap between data and a template model for use in matched filering as a function of time delay added to the template. Returns an array of time delays and their corresponding overlaps.
    data: data object
    model: model object
    """
    d = data.strain_fd
    h = model.strain_fd
    psd = data.psd
    T = data.T
    sig2 = T/4*psd
    Fs = data.Fs
    N = T*Fs
    ndet = d.shape[0]
    hh = np.array([inn_prod(h[i], h[i], psd[i], T) for i in range(ndet)])
    dh = np.array([ 1/2*N*np.fft.irfft( d[i]*np.conj(h[i])/sig2[i] ) - 1j/2*N*np.fft.irfft( 1j*d[i]*np.conj(h[i])/sig2[i] ) for i in range(ndet)])
    t = np.linspace(0, T-1/Fs, N)
    return t, np.sum(np.abs(dh)**2/hh[:, np.newaxis], axis=0)

def find_tc(data, model):
    """
    Finds the time delay for the template that maximizes the overlap with the data using the function time_delay_overlap.
    data: data object
    model: model object
    """
    t, overlap = time_delay_overlap(data, model)
    return t[np.argmax(overlap)]

def whiten_fd(strain_fd, psd, Fs):
    """
    Computes the whitened time series from a frequency series.
    strain_fd: frequency series array
    psd: psd array for strain_fd
    Fs: sampling rate (Hz)
    """
    return Fs*np.fft.irfft(np.sqrt(2/Fs/psd)*strain_fd)
    
def compute_exact_log_likelihood(data, model):
    """
    Computes the exact log likelihood for a given model and data.
    data: data object
    model: model object
    """
    T = data.T
    psd = data.psd
    h = model.strain_fd
    d = data.strain_fd
    ndet = data.ndet
    
    lnL = np.sum([inn_prod(d[i], h[i], psd[i], T) - 1/2*inn_prod(h[i], h[i], psd[i], T) for i in range(ndet)])
    return lnL

# Relative binning:

def get_binning_data_orig(fmin, fmax, f, delta=0.03):
    """
    Computes the binning data needed to do the original relative binning scheme
    (fbin: array of frequencies of the bin edges (Hz)
     Nbin: number of bins
     fbin_ind: array of indices of bin edges in frequency array f
     fm: frequencies of the center of the bins (Hz))
    using the standard analytic differential phase bounding (ADPB) method
    fmin: minimum frequency for bins (Hz)
    fmax: maximum frequency for the bins (Hz)
    f: full frequency array (Hz)
    delta: maximum allowed differential phase
    """
    # much of the code comes from gwpe.py
    f_nt = np.linspace(fmin, fmax, 10000)
    ga = np.array([-5/3, -2/3, 1, 5/3, 7/3])  # f^ga power law index
    dalp = 2 * np.pi / np.abs(fmin**ga - fmax**ga)
    dphi = np.sum([np.sign(g) * d * f_nt**g for d, g in zip(dalp, ga)], axis=0)
    dphi -= dphi[0]  # Worst case scenario differential phase

    # Now construct frequency bins
    nbin = int(np.ceil(dphi[-1] / delta))
    dphi_grid = np.linspace(dphi[0], dphi[-1], nbin + 1)
    fbin = np.interp(dphi_grid, dphi, f_nt)

    # Make sure grid points are on the FFT array:
    fbin_ind = np.unique(
        np.argmin(np.abs(f[:, np.newaxis] - fbin), axis=0))
    fbin = f[fbin_ind]  # Bin edges
    Nbin = len(fbin) - 1
    fm = 0.5*(fbin[1:]+fbin[:-1])
    
    return fbin, Nbin, fbin_ind, fm

def compute_modeless_summary_data(d, h0, psd, T, f, fbin, Nbin, fbin_ind, fm):
    """
    Computes the summary data (defined in equations 3-6 of arxiv.org/pdf/1806.08792.pdf) used in standard relative binning.
    d: data frequency series array
    h0: fiducial model frequency series array
    psd: psd array
    T: length of time series (s)
    f: full frequency array (Hz)
    fbin: array of frequencies of the bin edges (Hz)
    Nbin: number of bins
    fbin_ind: array of indices of bin edges in frequency array f
    fm: frequencies of the center of the bins (Hz)
    """
    ndet = h0.shape[0]
    
    A0 = 4/T*np.array([[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for i in range(ndet)])
    A1 = 4/T*np.array([[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for i in range(ndet)])
    B0 = 4/T*np.array([[np.sum(np.abs(h0[i, fbin_ind[b]:fbin_ind[b+1]])**2/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for i in range(ndet)])
    B1 = 4/T*np.array([[np.sum(np.abs(h0[i, fbin_ind[b]:fbin_ind[b+1]])**2/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for i in range(ndet)])

    return A0, A1, B0, B1

def compute_summary_data_1(d, hhat0, psd, T, f, fbin, Nbin, fbin_ind, fm):
    """
    Computes the summary data used in mode-by-mode relative binning (Scheme 1).
    d: data frequency series array
    hhat0: fiducial model time-dependent L-frame mode frequency series array
    psd: psd array
    T: length of time series (s)
    f: full frequency array (Hz)
    fbin: array of frequencies of the bin edges (Hz)
    Nbin: number of bins
    fbin_ind: array of indices of bin edges in frequency array f
    fm: frequencies of the center of the bins (Hz)
    """
    ndet = hhat0.shape[0]
    Nmode = hhat0.shape[1]
    
    A0 = 4/T*np.array([[[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(hhat0[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
    A1 = 4/T*np.array([[[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(hhat0[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
    B0 = 4/T*np.array([[[[np.sum(hhat0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(hhat0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
    B1 = 4/T*np.array([[[[np.sum(hhat0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(hhat0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])

    return A0, A1, B0, B1

def compute_summary_data_2(d, h0, psd, T, f, fbin, Nbin, fbin_ind, fm):
    """
    Computes the summary data used in mode-by-mode relative binning (Scheme 2).
    d: data frequency series array
    h0: fiducial model frequency series array
    psd: psd array
    T: length of time series (s)
    f: full frequency array (Hz)
    fbin: array of frequencies of the bin edges (Hz)
    Nbin: number of bins
    fbin_ind: array of indices of bin edges in frequency array f
    fm: frequencies of the center of the bins (Hz)
    """
    ndet = h0.shape[0]
    Nmode = h0.shape[1]
    
    A0 = 4/T*np.array([[[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
    A1 = 4/T*np.array([[[np.sum(d[i, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, l, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for i in range(ndet)])
    B0 = 4/T*np.array([[[[np.sum(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])
    B1 = 4/T*np.array([[[[np.sum(h0[i, l, fbin_ind[b]:fbin_ind[b+1]]*np.conj(h0[i, ll, fbin_ind[b]:fbin_ind[b+1]])/psd[i, fbin_ind[b]:fbin_ind[b+1]]*(f[fbin_ind[b]:fbin_ind[b+1]]-fm[b])) for b in range(Nbin)] for l in range(Nmode)] for ll in range(Nmode)] for i in range(ndet)])

    return A0, A1, B0, B1

def compute_modeless_bin_coefficients(fbin, r):
    """
    Computes the bin coefficients (defined in equation 1 of arxiv.org/pdf/1806.08792.pdf) used in standard modeless relative binning.
    fbin: array of frequencies of the bin edges (Hz)
    r: array of ratios of model over fiducial model, indices: detector, frequency
    """
    binwidths = fbin[1:] - fbin[:-1]
    
    # rplus: right edges of bins
    rplus = r[:, 1:]
    # rminus: left edges of bins
    rminus = r[:, :-1]
    
    r0 = 0.5 * (rplus + rminus)
    r1 = (rplus - rminus) / binwidths[np.newaxis, :]
    
    return r0, r1

def compute_bin_coefficients(fbin, r):
    """
    Computes the bin coefficients used in mode-by-mode relative binning.
    fbin: array of frequencies of the bin edges (Hz)
    r: array of ratios of model over fiducial model, indices: detector, mode, frequency
    """
    binwidths = fbin[1:] - fbin[:-1]
    
    # rplus: right edges of bins
    rplus = r[:, :, 1:]
    # rminus: left edges of bins
    rminus = r[:, :, :-1]
    
    r0 = 0.5 * (rplus + rminus)
    r1 = (rplus - rminus) / binwidths[np.newaxis, np.newaxis, :]
    
    return r0, r1
    
def compute_modeless_overlaps(r0, r1, A0, A1, B0, B1):
    """
    Compute the overlaps (as in equation 7 of arxiv.org/pdf/1806.08792.pdf) used in standard modeless relative binning.
    r0, r1: linear coefficients for the ratio of model over fiducial model, indices: detector, frequency
    A0, A1, B0, B1: summary data for relative binning, indices, detector, frequency
    """
    
    Zdh = np.real(np.sum( A0*np.conj(r0) + A1*np.conj(r1) ))
    
    Zhh = np.real(np.sum( B0*np.abs(r0)**2 + 2*B1*np.real( r0*np.conj(r1))))
    
    return Zdh, Zhh

def compute_overlaps_1(r0, r1, C0, C1, A0, A1, B0, B1):
    """
    Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf) used in advanced mode by mode relative binning (Scheme 1).
    r0, r1: linear coefficients for the ratio of time-dependent L-frame modes over fiducial time-dependent L-frame modes, indices: detector, mode, frequency
    C0, C1: linear coefficients for the coefficient of the time-dependent L-frame modes, indices: detector, mode, frequency
    A0, A1, B0, B1: summary data for mode-by mode relative binning, A indices: detector, mode, frequency, B indices: detector, mode, mode, frequency
    """
    
    Zdh = np.real(np.sum( A0*np.conj(r0)*np.conj(C0) 
                        + A1*( np.conj(r0)*np.conj(C1) + np.conj(r1)*np.conj(C0) ) ))
    
    Zhh = np.real(np.sum( B0*r0[:, np.newaxis, :, :]*C0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*np.conj(C0[:, :, np.newaxis, :]) 
                        + B1*( r0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :])*(np.conj(C0[:, :, np.newaxis, :])*C1[:, np.newaxis, :, :] + C0[:, np.newaxis, :, :]*np.conj(C1[:, :, np.newaxis, :])) + C0[:, np.newaxis, :, :]*np.conj(C0[:, :, np.newaxis, :])*(np.conj(r0[:, :, np.newaxis, :])*r1[:, np.newaxis, :, :] + r0[:, np.newaxis, :, :]*np.conj(r1[:, :, np.newaxis, :])) ) ))
    
    return Zdh, Zhh

def compute_overlaps_2(r0, r1, A0, A1, B0, B1):
    """
    Compute the overlaps (similar to equation 7 of arxiv.org/pdf/1806.08792.pdf) used in mode by mode relative binning (Scheme 2).
    r0, r1: linear coefficients for the ratio of model over fiducial model, indices: detector, mode, frequency
    A0, A1, B0, B1: summary data for mode-by mode relative binning, A indices: detector, mode, frequency, B indices: detector, mode, mode, frequency
    """
    
    Zdh = np.real(np.sum( A0*np.conj(r0) + A1*np.conj(r1) ))
    
    Zhh = np.real(np.sum( B0*r0[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :]) + B1*( r0[:, np.newaxis, :, :]*np.conj(r1[:, :, np.newaxis, :]) + r1[:, np.newaxis, :, :]*np.conj(r0[:, :, np.newaxis, :]) ) ))
    
    return Zdh, Zhh

def setup_binning(data, fiducial_model, modebymode = True, test_model = None, Etot = 0.01, correlated_bin_error = True, rmax = 2, scheme = 1):
    """
    Compute the binning data for relative binning. This only needs to be done once for a particular fiducial model. The results are used in computing the likelihood with relative binning for a given model.
    data: data object
    fiducial_model: fiducial model object, must have compute_modes = True
    modebymode: boolean, if True: computes the binning data for relative binning mode by mode
                           if False: computes the binning data for standard modeless relative binning
    test_model: test model object required for bisecting bin selection algorithm
    Etot: total error target for bisecting bin selection algorithm
    correlated_bin_error: boolean, used in bisecting bin selection algorithm, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                                        if False: set bound on error per bin assuming all bin errors are uncorrelated
    scheme: an int that specifies the scheme to be used, acceptable values are 1 and 2, 1 is used by default
    """
    T = data.T
    Fs = data.Fs
    f = data.fs
    psd = data.psd
    d = data.strain_fd
    fmin = fiducial_model.fmin
    fmax = fiducial_model.fmax
    
    if modebymode:
        if test_model == None:
            raise ValueError('Test model required for mode by mode relative binning.')
        if scheme == 1:
            fiducial_model.compute_td_L_frame_modes()
            hhat0 = fiducial_model.td_L_frame_modes
            test_model.compute_td_L_frame_modes()
            hhat = test_model.td_L_frame_modes
            test_model.compute_C_coefficients()
            C = test_model.Ccoeffs
            fbin, Nbin, fbin_ind, fm = get_binning_data_bisect_1(data, hhat0, hhat, C, fmin, fmax, 200, Etot, correlated_bin_error = correlated_bin_error)
            
            hhat0_fbin = hhat0[:, :, fbin_ind]

            A0, A1, B0, B1 = compute_summary_data_1(d, hhat0, psd, T, f, fbin, Nbin, fbin_ind, fm)
            return fbin, hhat0_fbin, A0, A1, B0, B1

        elif scheme == 2:
            fiducial_model.compute_modes()
            h0 = fiducial_model.modes
            test_model.compute_modes()
            h = test_model.modes
            fbin, Nbin, fbin_ind, fm = get_binning_data_bisect_2(data, h0, h, fmin, fmax, 200, Etot, correlated_bin_error = correlated_bin_error)

            A0, A1, B0, B1 = compute_summary_data_2(d, h0, psd, T, f, fbin, Nbin, fbin_ind, fm)
            h0_fbin = h0[:, :, fbin_ind]
            return fbin, h0_fbin, A0, A1, B0, B1

        else:
            raise ValueError('The variable \'scheme\' must be either the int 1 or the int 2.')
    else:
        fbin, Nbin, fbin_ind, fm = get_binning_data_orig(fmin, fmax, f)
        h0 = fiducial_model.strain_fd

        A0, A1, B0, B1 = compute_modeless_summary_data(d, h0, psd, T, f, fbin, Nbin, fbin_ind, fm)
        h0_fbin = h0[:, fbin_ind]
    
    return fbin, h0_fbin, A0, A1, B0, B1
    

def compute_modeless_relative_binning_log_likelihood(binning_info, pdict, uselal=False, detstrings = ['H1', 'L1', 'V1']):
    """
    Compute the log likelihood using standard modeless relative binning.
    binning_info: tuple, output by setup_binning with mode_by_mode = False, contains fbin: array of frequencies of the bin edges (Hz)
                                                                                     h0_fbin: fiducial model evaluated at the bin edges
                                                                                     A0, A1, B0, B1: summary data for modeless relative binning
    
    pdict: parameter dictionary for the model evaluation
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    fbin, h0, A0, A1, B0, B1 = binning_info
    fbin_seq = lal.CreateREAL8Sequence(len(fbin))
    fbin_seq.data = fbin
    h = xphm.compute_strain(pdict, fbin_seq, uselal = uselal, detstrings=detstrings)
    
    r = (h/h0)
    
    r0, r1 = compute_modeless_bin_coefficients(fbin, r)
    
    Zdh, Zhh = compute_modeless_overlaps(r0, r1, A0, A1, B0, B1)
    
    return Zdh - 1/2*Zhh



def compute_relative_binning_log_likelihood(binning_info, pdict, detstrings = ['H1', 'L1', 'V1'], scheme = 1):
    """
    Compute the log likelihood using mode-by-mode relative binning.
    binning_info: tuple, output by setup_binning with mode_by_mode = True, contains fbin: array of frequencies of the bin edges (Hz)
                                                                                    hhat0_fbin (scheme 1)/h0_fbin (scheme 2): fiducial model time dependent L-frame modes (scheme 1)/full strain mode components (scheme 2) evaluated at the bin edges
                                                                                    A0, A1, B0, B1: summary data for mode-by-mode relative binning
    
    pdict: parameter dictionary for the model evaluation
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    scheme: an int that specifies the scheme to be used, acceptable values are 1 and 2, 1 is used by default
    """
    if scheme == 1:
        fbin, hhat0, A0, A1, B0, B1 = binning_info
        fbin_seq = lal.CreateREAL8Sequence(len(fbin))
        fbin_seq.data = fbin
        hhat = xphm.compute_td_L_frame_modes(pdict, fbin_seq, detstrings=detstrings)
        C = xphm.compute_C_prefactors(pdict, fbin_seq, detstrings=detstrings)

        r = hhat/hhat0

        r0, r1 = compute_bin_coefficients(fbin, r)
        C0, C1 = compute_bin_coefficients(fbin, C)
        
        Zdh, Zhh = compute_overlaps_1(r0, r1, C0, C1, A0, A1, B0, B1)

        return Zdh - 1/2*Zhh
    
    elif scheme == 2:
        fbin, h0, A0, A1, B0, B1 = binning_info
        fbin_seq = lal.CreateREAL8Sequence(len(fbin))
        fbin_seq.data = fbin
        h = xphm.compute_strain_components(pdict, fbin_seq, detstrings=detstrings)

        r = (h/h0)

        r0, r1 = compute_bin_coefficients(fbin, r)

        Zdh, Zhh = compute_overlaps_2(r0, r1, A0, A1, B0, B1)

        return Zdh - 1/2*Zhh
    
    else:
        raise ValueError('The variable \'scheme\' must be either the int 1 or the int 2.')


# new bin selection algorithm
    
def compute_bin_error_1(data, hhat0, hhat, C, f_lo, f_hi, sign = False):
    """
    Computes the error contribution to the log likelihood of amode-by-mode relative binning scheme 1 for one bin that begins at the frequency in the data object closest to f_lo and ends at the frequency in the data object closest to f_hi.
    data: data object
    hhat0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
    hhat: test model time dependent L-frame modes, indices: detector, mode, frequency
    C: test model time dependent L-frame mode Coefficients, indices: detector, mode, frequency
    f_lo: target lower frequency for the bin, this exact frequency may not be in the frequency array
    f_hi: target upper frequency for the bin, this exact frequency may not be in the frequency array
    sign: boolean, if True: returns the error (relative binning - exact) with its sign, this option is used for investigating error in relative binning
                   if False: returns the absolute value of the error, this option is used in the bisecting bin selection algorithm
    """
    # build length 2 array for small bin
    d = data.strain_fd
    T = data.T
    psd = data.psd
    f = data.fs
    ndet = data.ndet
    fbin = np.array([f_lo, f_hi])
    fbin_ind = np.unique(
        np.argmin(np.abs(f[:, np.newaxis] - fbin), axis=0))
    fbin = f[fbin_ind]
    Nbin = 1
    fm = 0.5*(fbin[1:]+fbin[:-1])
    
    # compute relative binning log likelihood contribution for the given bin
    A0, A1, B0, B1 = compute_summary_data_1(d, hhat0, psd, T, f, fbin, Nbin, fbin_ind, fm)
    
    r = (hhat/hhat0)[:,:,fbin_ind]
    Cfbin = C[:,:,fbin_ind]
    
    r0, r1 = compute_bin_coefficients(fbin, r)
    C0, C1 = compute_bin_coefficients(fbin, Cfbin)
    
    Zdh, Zhh = compute_overlaps_1(r0, r1, C0, C1, A0, A1, B0, B1)
    
    rb_lnL = Zdh - 1/2*Zhh
    
    h_allmodes = np.sum(C*hhat, axis = 1)
    
    # compute exact log likelihood contribution
    exact_lnL = np.sum([inn_prod(d[i, fbin_ind[0]:fbin_ind[1]], h_allmodes[i, fbin_ind[0]:fbin_ind[1]], psd[i, fbin_ind[0]:fbin_ind[1]], T) - 1/2*inn_prod(h_allmodes[i, fbin_ind[0]:fbin_ind[1]], h_allmodes[i, fbin_ind[0]:fbin_ind[1]], psd[i, fbin_ind[0]:fbin_ind[1]], T) for i in range(ndet)])
    
    if sign:
        return rb_lnL - exact_lnL
    else:
        return np.abs(rb_lnL - exact_lnL)


def compute_bin_error_2(data, h0, h, f_lo, f_hi, sign = False):
    """
    Computes the error contribution to the log likelihood of mode-by-mode relative binning scheme 2 for one bin that begins at the frequency in the data object closest to f_lo and ends at the frequency in the data object closest to f_hi.
    data: data object
    h0: fiducial model modes, indices: detector, mode, frequency
    h: test model modes, indices: detector, mode, frequency
    f_lo: target lower frequency for the bin, this exact frequency may not be in the frequency array
    f_hi: target upper frequency for the bin, this exact frequency may not be in the frequency array
    sign: boolean, if True: returns the error (relative binning - exact) with its sign, this option is used for investigating error in relative binning
                   if False: returns the absolute value of the error, this option is used in the bisecting bin selection algorithm
    """
    # build length 2 array for small bin
    d = data.strain_fd
    T = data.T
    psd = data.psd
    f = data.fs
    ndet = data.ndet
    fbin = np.array([f_lo, f_hi])
    fbin_ind = np.unique(
        np.argmin(np.abs(f[:, np.newaxis] - fbin), axis=0))
    fbin = f[fbin_ind]
    Nbin = 1
    fm = 0.5*(fbin[1:]+fbin[:-1])
    
    # compute relative binning log likelihood contribution for the given bin
    A0, A1, B0, B1 = compute_summary_data_2(d, h0, psd, T, f, fbin, Nbin, fbin_ind, fm)
    
    r = (h/h0)[:,:,fbin_ind]
    
    r0, r1 = compute_bin_coefficients(fbin, r)
    
    Zdh, Zhh = compute_overlaps_2(r0, r1, A0, A1, B0, B1)
    
    rb_lnL = Zdh - 1/2*Zhh
    
    h_allmodes = np.sum(h, axis = 1)
    
    # compute exact log likelihood contribution
    exact_lnL = np.sum([inn_prod(d[i, fbin_ind[0]:fbin_ind[1]], h_allmodes[i, fbin_ind[0]:fbin_ind[1]], psd[i, fbin_ind[0]:fbin_ind[1]], T) - 1/2*inn_prod(h_allmodes[i, fbin_ind[0]:fbin_ind[1]], h_allmodes[i, fbin_ind[0]:fbin_ind[1]], psd[i, fbin_ind[0]:fbin_ind[1]], T) for i in range(ndet)])
    
    if sign:
        return rb_lnL - exact_lnL
    else:
        return np.abs(rb_lnL - exact_lnL)
    
    
def nearest_freq(f, target_f):
    """
    Returns the frequency in f closest to target_f.
    f: frequency array
    target_f: target frequency
    """
    f_ind = np.argmin(np.abs(f-target_f))
    f_near = f[f_ind]
    return f_near, f_ind

# bin selection algorithm for scheme 1
    
def get_binning_data_bisect_1(data, hhat0, hhat, C, fmin, fmax, N0, E, correlated_bin_error = True):
    """
    Computes the binning data needed to do mode by mode relative binning with scheme 1
    (fbin: array of frequencies of the bin edges (Hz)
     Nbin: number of bins
     fbin_ind: array of indices of bin edges in frequency array f
     fm: frequencies of the center of the bins (Hz))
    using the bisecting bin selection algorithm. This algorithm divides the region in 2 repeatedly, accepting bins when their error is below a threshold that achieves an overall target error for the test model. This algorithm iterates, changing the target number of bins until the threshold is achieved with the target number of bins.
    data: data object
    hhat0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
    hhat: test model time dependent L-frame modes, indices: detector, mode, frequency
    C: test model time dependent L-frame mode Coefficients, indices: detector, mode, frequency
    fmin: minimum frequency for bins (Hz)
    fmax: maximum frequency for the bins (Hz)
    N0: initial target number of bins
    E: total target error
    correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                   if False: set bound on error per bin assuming all bin errors are uncorrelated
    """
    fbin, Nbin, fbin_ind, fm = get_binning_data_bisect_1_iteration(data, hhat0, hhat, C, fmin, fmax, N0, E, correlated_bin_error = correlated_bin_error)
    if Nbin == N0:
        return fbin, Nbin, fbin_ind, fm
    else:
        return get_binning_data_bisect_1(data, hhat0, hhat, C, fmin, fmax, Nbin, E, correlated_bin_error = correlated_bin_error)

def get_binning_data_bisect_1_iteration(data, hhat0, hhat, C, fmin, fmax, N, E, correlated_bin_error):
    """
    Helper function for get_binning_data_bisect_1. This function does bisection checks trying to achieve the target error E in N bins. It may require more bins or fewer bins. If so, this function is called again.
    data: data object
    hhat0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
    hhat: test model time dependent L-frame modes, indices: detector, mode, frequency
    C: test model time dependent L-frame mode Coefficients, indices: detector, mode, frequency
    fmin: minimum frequency for bins (Hz)
    fmax: maximum frequency for the bins (Hz)
    N: target number of bins
    E: total target error
    correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                   if False: set bound on error per bin assuming all bin errors are uncorrelated
    """
    f = data.fs
    fmin, fmin_ind = nearest_freq(f, fmin)
    fmax, fmax_ind = nearest_freq(f, fmax)
    max_bisections = np.floor(np.log2(fmax_ind-fmin_ind))
    if correlated_bin_error:
        fbin_nomin, Nbin, fbin_ind_nomin = bisect_bin_search_1(data, hhat0, hhat, C, fmin, fmax, 0, max_bisections, E/N) #worst case error addition, assuming strong correlation
    else:
        fbin_nomin, Nbin, fbin_ind_nomin = bisect_bin_search_1(data, hhat0, hhat, C, fmin, fmax, 0, max_bisections, E/np.sqrt(N)) #assuming errors are independent, seems roughly correct from testing
    fbin = np.append(fmin, fbin_nomin)
    fbin_ind = np.append(fmin_ind, fbin_ind_nomin)
    fm = (fbin[1:] + fbin[:-1])/2
    return fbin, Nbin, fbin_ind, fm

def bisect_bin_search_1(data, hhat0, hhat, C, f_lo, f_hi, depth, maxdepth, Emax):
    """
    Helper function for get_binning_data_bisect_1_iteration. This recursive function does the bisection until either the target is achieved or the maximum bisection depth is reached.
    data: data object
    hhat0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
    hhat: test model time dependent L-frame modes, indices: detector, mode, frequency
    C: test model time dependent L-frame mode Coefficients, indices: detector, mode, frequency
    f_lo: lower frequency for a test bin (Hz)
    f_hi: upper frequency for a test bin (Hz)
    depth: number of times bisection has been performed
    maxdepth: maximum number of times bisection is to be performed before the bisection stops
    Emax: maximum error allowed per bin (unless maxdepth is reached)
    """
    if depth == maxdepth:
        fmax, fmax_ind = nearest_freq(data.fs, f_hi)
        return np.array([fmax]), 1, np.array([fmax_ind])
    bin_error = compute_bin_error_1(data, hhat0, hhat, C, f_lo, f_hi)
    if bin_error < Emax:
        fmax, fmax_ind = nearest_freq(data.fs, f_hi)
        return np.array([fmax]), 1, np.array([fmax_ind])
    else:
        f_mid = (f_lo + f_hi)/2
        fbin_lo, Nbin_lo, fbin_ind_lo = bisect_bin_search_1(data, hhat0, hhat, C, f_lo, f_mid, depth+1, maxdepth, Emax)
        fbin_hi, Nbin_hi, fbin_ind_hi = bisect_bin_search_1(data, hhat0, hhat, C, f_mid, f_hi, depth+1, maxdepth, Emax)
        fbin = np.append(fbin_lo, fbin_hi)
        Nbin = Nbin_lo + Nbin_hi
        fbin_ind = np.append(fbin_ind_lo, fbin_ind_hi)
        return fbin, Nbin, fbin_ind
    
# bin selection algorithm for scheme 2

def get_binning_data_bisect_2(data, h0, h, fmin, fmax, N0, E, correlated_bin_error = True):
    """
    Computes the binning data needed to do mode by mode relative binning with scheme 2
    (fbin: array of frequencies of the bin edges (Hz)
     Nbin: number of bins
     fbin_ind: array of indices of bin edges in frequency array f
     fm: frequencies of the center of the bins (Hz))
    using the bisecting bin selection algorithm. This algorithm divides the region in 2 repeatedly, accepting bins when their error is below a threshold that achieves an overall target error for the test model. This algorithm iterates, changing the target number of bins until the threshold is achieved with the target number of bins.
    data: data object
    h0: fiducial model time dependent L-frame modes, indices: detector, mode, frequency
    h: test model modes, indices: detector, mode, frequency
    fmin: minimum frequency for bins (Hz)
    fmax: maximum frequency for the bins (Hz)
    N0: initial target number of bins
    E: total target error
    correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                   if False: set bound on error per bin assuming all bin errors are uncorrelated
    """
    fbin, Nbin, fbin_ind, fm = get_binning_data_bisect_2_iteration(data, h0, h, fmin, fmax, N0, E, correlated_bin_error = correlated_bin_error)
    if Nbin == N0:
        return fbin, Nbin, fbin_ind, fm
    else:
        return get_binning_data_bisect_2(data, h0, h, fmin, fmax, Nbin, E, correlated_bin_error = correlated_bin_error)

def get_binning_data_bisect_2_iteration(data, h0, h, fmin, fmax, N, E, correlated_bin_error):
    """
    Helper function for get_binning_data_bisect_2. This function does bisection checks trying to achieve the target error E in N bins. It may require more bins or fewer bins. If so, this function is called again.
    data: data object
    h0: fiducial model modes, indices: detector, mode, frequency
    h: test model modes, indices: detector, mode, frequency
    fmin: minimum frequency for bins (Hz)
    fmax: maximum frequency for the bins (Hz)
    N: target number of bins
    E: total target error
    correlated_bin_error: boolean, if True: set bound on error per bin assuming the worst case: the bin errors all add
                                   if False: set bound on error per bin assuming all bin errors are uncorrelated
    """
    f = data.fs
    fmin, fmin_ind = nearest_freq(f, fmin)
    fmax, fmax_ind = nearest_freq(f, fmax)
    max_bisections = np.floor(np.log2(fmax_ind-fmin_ind))
    if correlated_bin_error:
        fbin_nomin, Nbin, fbin_ind_nomin = bisect_bin_search_2(data, h0, h, fmin, fmax, 0, max_bisections, E/N) #worst case error addition, assuming strong correlation
    else:
        fbin_nomin, Nbin, fbin_ind_nomin = bisect_bin_search_2(data, h0, h, fmin, fmax, 0, max_bisections, E/np.sqrt(N)) #assuming errors are independent, seems roughly correct from testing
    fbin = np.append(fmin, fbin_nomin)
    fbin_ind = np.append(fmin_ind, fbin_ind_nomin)
    fm = (fbin[1:] + fbin[:-1])/2
    return fbin, Nbin, fbin_ind, fm

def bisect_bin_search_2(data, h0, h, f_lo, f_hi, depth, maxdepth, Emax):
    """
    Helper function for get_binning_data_bisect_2_iteration. This recursive function does the bisection until either the target is achieved or the maximum bisection depth is reached.
    data: data object
    h0: fiducial model modes, indices: detector, mode, frequency
    h: test model modes, indices: detector, mode, frequency
    f_lo: lower frequency for a test bin (Hz)
    f_hi: upper frequency for a test bin (Hz)
    depth: number of times bisection has been performed
    maxdepth: maximum number of times bisection is to be performed before the bisection stops
    Emax: maximum error allowed per bin (unless maxdepth is reached)
    """
    if depth == maxdepth:
        fmax, fmax_ind = nearest_freq(data.fs, f_hi)
        return np.array([fmax]), 1, np.array([fmax_ind])
    bin_error = compute_bin_error_2(data, h0, h, f_lo, f_hi)
    if bin_error < Emax:
        fmax, fmax_ind = nearest_freq(data.fs, f_hi)
        return np.array([fmax]), 1, np.array([fmax_ind])
    else:
        f_mid = (f_lo + f_hi)/2
        fbin_lo, Nbin_lo, fbin_ind_lo = bisect_bin_search_2(data, h0, h, f_lo, f_mid, depth+1, maxdepth, Emax)
        fbin_hi, Nbin_hi, fbin_ind_hi = bisect_bin_search_2(data, h0, h, f_mid, f_hi, depth+1, maxdepth, Emax)
        fbin = np.append(fbin_lo, fbin_hi)
        Nbin = Nbin_lo + Nbin_hi
        fbin_ind = np.append(fbin_ind_lo, fbin_ind_hi)
        return fbin, Nbin, fbin_ind