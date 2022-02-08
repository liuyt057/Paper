# -*- coding: utf-8 -*-
import numpy as np
from librosa import load
from spafe.utils.cepstral import cms, cmvn
from spafe.fbanks.mel_fbanks import mel_filter_banks
from spafe.utils.spectral import rfft
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
import matplotlib.pyplot as plt
import spafe.utils.vis as vis


# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate / 2, fft_size // 2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()


# 提取FBank特征函数
def MyFBank(sig,
            fs=16000,
            pre_emph=1,
            win_len=0.025,
            win_hop=0.01,
            win_type="hamming",
            nfilts=26,
            nfft=512,
            low_freq=None,
            high_freq=None,
            scale="constant",
            normalize=1):
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = (1 / 1) * np.abs(fourrier_transform)

    #  -> x Mel-fbanks
    mel_fbanks_mat = mel_filter_banks(nfilts=nfilts,
                                      nfft=nfft,
                                      fs=fs,
                                      low_freq=low_freq,
                                      high_freq=high_freq,
                                      scale=scale)
    features = np.dot(abs_fft_values, mel_fbanks_mat.T)

    # -> log(.)
    features_no_zero = zero_handling(features)
    myFBanks = np.log(features_no_zero)

    # normalizatio
    if normalize:
        myFBanks = cmvn(cms(myFBanks))

    return myFBanks


if __name__ == "__main__":
    signal, sample_rate = load("D:/Research/ShipsEar/data/train_wavdata/0/46__23_07_13_H3_bateero1.wav")
    signal = signal[0: int(0.34 * sample_rate)]

    # 计算声音特征
    FBanks = MyFBank(signal, fs=sample_rate, nfilts=32)
    print(FBanks.shape)
    # 特征可视化
    vis.visualize_features(FBanks, "Mel Filter Banks", 'Frame Index')
