import os
import librosa
import numpy as np
from tqdm import tqdm
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as aF

window_length = (50 * 1e-3)
hop_length = (50 * 1e-3)


def load_wav(filename):
    """Rea audio file and return audio signal and sampling frequency"""
    if not os.path.exists(filename):
        raise FileNotFoundError
    # Load file using librosa
    x, fs = librosa.load(filename, sr=None)
    return x, fs


def melspectrogram(x=None, fs=None, n_fft=None, hop_length=None,
                       fuse=False):
    """Returns a mel spectrogram."""

    if x is None:
        return None
    # Set some values
    if n_fft is None:
        n_fft = int(window_length * fs)
    if hop_length is None:
        hop_length = int(hop_length * fs)
    # Get spectrogram
    spectrogram = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=n_fft,
                                                 hop_length=hop_length)
    # Convert to MEL-Scale
    spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)  # (n_mel,t)

    if fuse:
        chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=n_fft,
                                             hop_length=hop_length)
        chroma_dB = librosa.power_to_db(chroma)
        out = np.concatenate((spectrogram_dB.T, chroma_dB.T), axis=1)
    else:
        # Transpose to return (time,n_mel)
        out = spectrogram_dB.T
    return out


def get_melgram(file):
    signal, fs = load_wav(file)
    melgram = melspectrogram(
        signal, fs=fs, n_fft=int(window_length * fs), hop_length=int(hop_length * fs), fuse=False)
    return melgram


def pyaudio_read(filenames):
    """Read file using pyAudioAnalysis"""

    #Consider same sampling frequencies
    sequences = []
    for file in filenames:
        fs, samples = aIO.read_audio_file(file)
        sequences.append(samples)

    sequences = np.asarray(sequences)

    return sequences, fs


def pyaudio_segment_features(filenames):
    
    segment_features_all = []
    segment_features_stats_all = []

    sequences, sampling_rate = pyaudio_read(filenames)

    mid_window = 3
    mid_step = 1
    short_window = 0.05
    short_step = 0.05

    for seq in tqdm(sequences):
        (segment_features_stats, segment_features,
         feature_names) = aF.mid_feature_extraction(
            seq, sampling_rate, round(mid_window * sampling_rate),
            round(mid_step * sampling_rate),
            round(sampling_rate * short_window),
            round(sampling_rate * short_step))

        segment_features_all.append(np.asarray(segment_features))
        
        segment_features_stats = np.asarray(segment_features_stats)
        segment_features_stats_all.append(segment_features_stats)

    return segment_features_all, segment_features_stats_all, feature_names

