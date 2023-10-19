import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from okr.utils import normalize







def load_audio(example):
    # loads an audio file and normalizes it to 0 mean and 1 std.
    audio, sr = sf.read(example['audio_path'])
    audio = normalize(audio)
    example['audio_data'] = audio
    return example

def fix_length(example):
    # Trims or pads examples to 1 second long
    audio = example['audio_data']
    if len(audio) >= 8192 * 2:
        audio = audio[:8192 * 2]
    else:
        audio = np.pad(audio, pad_width=(0, 8192 * 2 - len(audio)))
    example['audio_data'] = audio
    return example

def log_mel_wrapper(example):
    # Wrapped around the Mel_spectrograms class to work with Lazy_dataset.
    example['mels'] = torch.from_numpy(Mel_spectrograms()._ln(example['audio_data']).T)
    return example


class Mel_spectrograms():
    """
        Calculates mel spectrograms given an audio input

        Args:
            audio (ndarray): Audio signal (T,).
            sr (int): Sampling rate.
            fft_size (int): The size of the fft.
            hop_size (int): The shift per bin.
            win_length (int): Window length. If set to None, it will be equal to fft_size.
            window (str): Window function type.
            num_mels (int): Number of mel basis.
            fmin (int): Minimum frequency in mel basis calculation.
            fmax (int): Maximum frequency in mel basis calculation.
            eps (float): Epsilon value to avoid inf in log calculation.
        Returns:
            ndarray: Log Mel filterbank feature (#frames, num_mels).   

    """
    def __init__(self,sr=16000, fft_size=1024,hop_size=256, win_length=1024,window='hann', num_mels=80, fmin=0,fmax=8000, eps=1e-8):
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window =window
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps



    def get_mels(self, audio):
        # get amplitude spectrogram
        x_stft = librosa.stft(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            pad_mode="reflect",
        )
        spc = np.abs(x_stft).T  # (#frames, #bins)

        # get mel basis

        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.fft_size,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        mel = np.maximum(self.eps, np.dot(spc, mel_basis.T))[:-1,:]
        return mel
    def _log2(self, audio):
        return np.log2(self.get_mels(audio)).astype(np.float32)
    def _log10(self, audio):
        return np.log10(self.get_mels(audio)).astype(np.float32)
    def _ln(self, audio):
        return np.log(self.get_mels(audio)).astype(np.float32)



def collate(example):
    # transforms a list of dictionaries to a dictionary of lists, sharing the same keys
    temp_dict = dict()
    for item in example:
        for key in item:
            if key not in temp_dict.keys():
                temp_dict[key] = []
            temp_dict[key].append(item[key])
    example = temp_dict
    return example




def plot_feature(feature):
    fig, ax = plt.subplots()
    im = ax.imshow(feature, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig







