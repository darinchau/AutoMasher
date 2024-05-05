import numpy as np


def band_limited_noise(min_freq: float = 16.35, max_freq: float = 4978.03, samples: int = 1024, sample_rate=1):
    """
        This function can be used for generating random noise for
        stationary denoising.

        Param:
            min_freq: minimum frequency, the default value is the hertz value of C0
            max_freq: maximum frequencey
            samples: number of samples to be generate
            sample_rate: the sample rate of the sample
        
        Return:
            Generated random noise
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1 / sample_rate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real