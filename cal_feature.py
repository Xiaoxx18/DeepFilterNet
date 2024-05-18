import torch.nn as nn
import numpy as np
import math
import torch

class ERB(nn.Module):
    def __init__(self, nb_df, sr, fft_size, hop_size, nb_bands, min_nb_freqs, tau):
        self.nb_df = nb_df
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_bands = nb_bands
        self.min_nb_freqs = min_nb_freqs
        self.tau = tau


        self.a = self.get_norm_alpha(sr, hop_size, tau)
        self.widths = self.erb_width(sr, fft_size, 0, sr / 2, nb_bands, min_nb_freqs)
        self.erb_m = self.get_erb_m(self.widths, sr, inverse=False)
        self.erb_weights = self.erb_fb(self.widths, sr, inverse=False)
        self.erb_inverse_weight = self.erb_fb(self.widths, sr, inverse=True)

    def freq2erb(self, freq_hz):
        return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
    
    def erb2freq(self, n_erb):
        return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)
    
    def _calculate_norm_alpha(self, sr, hop_size, tau):
        dt = hop_size / sr
        return math.exp(-dt / tau)
    
    def get_norm_alpha(self, sr, hop_size, tau):
        a_ = self._calculate_norm_alpha(sr, hop_size, tau=0.1)
        precision = 3
        a = 1.0
        while a >= 1.0:
            a = round(a_, precision)
            precision += 1
        
        return a
    
    def erb_width(self, sr, fft_size, low_freq, high_freq, nb_bands, min_nb_freqs):
        freq_width = sr / fft_size
        erb_low = self.freq2erb(low_freq)
        erb_high = self.freq2erb(high_freq)
        erb_fb = np.zeros(nb_bands, dtype=np.unit8)
        step = (erb_high - erb_low) / nb_bands
        min_nb_freqs = min_nb_freqs

        prev_freq = 0
        freq_over = 0

        for i in range(1, nb_bands+1):
            f = self.erb2freq(erb_low + i * step)
            fb = round(f / freq_width)
            nb_freqs = fb - prev_freq - freq_over
            if nb_freqs < min_nb_freqs:
                freq_over = min_nb_freqs - nb_freqs
                nb_freqs = min_nb_freqs
            else:
                freq_over = 0
        
        erb_fb[nb_bands - 1] += 1
        too_large = erb_fb.sum() - (fft_size / 2 + 1)
        if too_large > 0:
            erb_fb[nb_bands - 1] -= too_large

        return torch.from_numpy(erb_fb)
    
    def get_erb_m(self, erb_fb, fft_size):
        erb_leng = len(erb_fb)
        erb_m = torch.zeros([fft_size, erb_leng], dtype=torch.float32)
        bcsum = 0
        for band_indx in range(len(erb_fb)):
            band_size = int(erb_fb[band_indx])
            k = 1. / band_size
            erb_m[bcsum:bcsum+band_size, band_indx] = k
            bcsum += band_size
        return erb_m
    
    def erb(self, input, erb_m, db=True):
        in_real = input(..., 0)
        in_imag = input(..., 1)
        output = (in_real ** 2 + in_imag ** 2)
        output = torch.matmul(output, erb_m)
        if db:
            output = 10 * torch.log10(output + 1e-10)

        return output
    
    def band_mean_norm_erb(self, xs, state, alpha):
        state = xs * (1. - alpha) + state * alpha
        xs = (xs - state) / 40.0

        return xs, state
    
    def erb_norm(self, input, alpha):
        b = input.shape[2]
        state_ch0 = torch.linspace(-60, -90, b, dtype=torch.float32)
        state = state_ch0

        norm_erb = torch.zeros(input.shape, device=torch.float32)
        for f_channel in range(input.shape[0]):
            for in_step in range(input.shape[1]):
                norm_erb[f_channel, in_step, :], state = self.band_mean_norm_erb(input[f_channel, in_step, :], state, alpha)
        return norm_erb
    
    def band_unit_norm(self, xs, state, alpha):
        xs_real = xs[..., 0]
        xs_imag = xs[..., 1]
        mag_abs = torch.sqrt(xs_real ** 2 + xs_imag ** 2)
        state = torch.sqrt(mag_abs * (1. - alpha) + state * alpha)
        xs_real = xs_real / state
        xs_imag = xs_imag / state
        xs = torch.stack([xs_real, xs_imag], dim=1)

        return xs, state
    
    def unit_norm(self, input, alpha):

        f = input.shape[2]
        state_ch0 = torch.linspace(0.001, 0.0001, f, dtype=torch.float32)
        state = state_ch0
        norm_unit = torch.zeros(input.shape, dtype=torch.float32)
        for f_channel in range(input.shape[0]):
            for in_step in range(input.shape[1]):
                norm_unit[f_channel, in_step, :], state = self.band_unit_norm(input[f_channel, in_step, :], state, alpha)

        return norm_unit
    
    def erb_fb(self, widths, sr, normalized=True, inverse=False):
        widths = widths.numpy().astype(np.float32)
        n_freqs = int(np.sum(widths))
        all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

        b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]
        fb = torch.zeros(all_freqs.shape[0], b_pts.shape[0])

        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = i
        
        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= fb.sum(dim=1, keepdim=True)
        else:
            if normalized:
                fb /= fb.sum(dim=0)
        
        return fb

    def forward(self, input):
        '''
        提取特征时，将代码写在这即可，也可以跟网络部分写成一块
        '''




 