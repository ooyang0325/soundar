from pydub import AudioSegment
from scipy.io import wavfile
from frange import *
import numpy as np
from const_value import frequency

dataset_path = "../dataset/"


def generate_silence():
    """Generate silence wav file padding 100 ms silence at the beginning of the file.

    (useless now)

    Usage:
        generate_silence()

    """
    # https://stackoverflow.com/questions/46757852/adding-silent-frame-to-wav-file-using-python

    for f in frequency:
        filename = f"../data/sin_{f}.wav"
        segment = AudioSegment.silent(duration=100)  # add 100 ms of silence
        wave = AudioSegment.from_wav(filename)
        wave = segment + wave
        wave.export(f"sin_{f}_delay.wav", format="wav")


def Efficient_ccf(x1, x2):
    """
    Reference: https://github.com/bingo-todd/Bianural-cues/blob/master/binaural_cues.ipynb

    calculate cross-crrelation function in frequency domain, which is more
    efficient than the direct calculation
    """

    if x1.shape[0] != x2.shape[0]:
        raise Exception("length mismatch")
    wav_len = x1.shape[0]
    # hanning window before fft
    wf = np.hanning(wav_len)
    x1 = x1 * wf
    x2 = x2 * wf

    X1 = np.fft.fft(x1, 2 * wav_len - 1)  # equivalent to add zeros
    X2 = np.fft.fft(x2, 2 * wav_len - 1)
    ccf_unshift = np.real(np.fft.ifft(np.multiply(X1, np.conjugate(X2))))
    ccf = np.concatenate([ccf_unshift[wav_len:], ccf_unshift[:wav_len]], axis=0)

    return ccf

def another_ccf(wav_data):
    interp = 16

    total_len = len(wav_data[:, 0]) + len(wav_data[:, 1])
    left_fft = np.fft.rfft(wav_data[:, 0], n=total_len)
    right_fft = np.fft.rfft(wav_data[:, 1], n=total_len)
    r = left_fft * np.conj(right_fft)

    cross_correlation = np.fft.irfft(r / np.abs(r), n=(total_len * interp))
    max_shift = int(interp * total_len / 2)
    cross_correlation = np.concatenate(
        (cross_correlation[-max_shift:], cross_correlation[: max_shift + 1])
    )
    return cross_correlation


def gcc_phat(dataset_path, dataset):

    ITD_list = []

    for data in iter(dataset):
        filename = data["filename"]
        sr, wav_readonly_data = wavfile.read(dataset_path + filename)
        wav_data = wav_readonly_data / 32767
        wav_len = wav_data.shape[0]
        max_delay = int(sr * 0.001)  # max delay for 1ms


        o = 2
        if o == 1:
            ccf_full = Efficient_ccf(wav_data[:, 0], wav_data[:, 1])
        elif o == 2:
            ccf_full = np.correlate(wav_data[:, 0], wav_data[:, 1], mode="full")
        elif o == 3:
            ccf_full = another_ccf(wav_data)
        ccf = ccf_full[wav_len - 1 - max_delay : wav_len + max_delay]

        ccf_std = ccf / (np.sqrt(np.sum(wav_data[:, 0] ** 2)) * np.sqrt(np.sum(wav_data[:, 1] ** 2)))
        max_pos = np.argmax(ccf)

        delta = 0
        if max_pos > 0 and max_pos < 2 * max_delay:
            delta = 0.5 * (ccf[max_pos - 1] - ccf[max_pos + 1]) / (ccf[max_pos + 1] - 2 * ccf[max_pos] + ccf[max_pos - 1])

        ITD = float(max_pos + delta - max_delay - 1) / sr * 1000
        ITD_list.append(ITD)

    print(ITD_list[:5])

    return ITD_list

def gcc_phat_kind_of_useless(dataset_path, dataset):

    result = []
    interp = 2

    for data in iter(dataset):
        filename = data["filename"]
        sr, wav_readonly_data = wavfile.read(dataset_path + filename)
        wav_data = wav_readonly_data / 32767

        total_len = len(wav_data[:, 0]) + len(wav_data[:, 1])
        left_fft = np.fft.rfft(wav_data[:, 0], n=total_len)
        right_fft = np.fft.rfft(wav_data[:, 1], n=total_len)
        r = left_fft * np.conj(right_fft)

        cross_correlation = np.fft.irfft(r / np.abs(r), n=(total_len * interp))
        max_shift = int(interp * total_len / 2)
        cross_correlation = np.concatenate(
            (cross_correlation[-max_shift:], cross_correlation[: max_shift + 1])
        )
        shift = np.argmax(np.abs(cross_correlation)) - max_shift

        tau = shift / float(interp * sr)

        print("tau theta", tau, data["theta"])
        result.append(tau)

    return result


def get_ILD(dataset_path, dataset):
    """ILD: Interaural Level Difference

    Usage:
        ILD_list = get_ILD(dataset_path, dataset)

    Args:
        dataset_path (string): path to dataset
        dataset (array of dict): dataset

    Returns:
        list of float: ILD List of dataset
    """

    print("get ILD")

    ILD_list = []

    for data in iter(dataset):
        filename = data["filename"]
        sr, wav_readonly_data = wavfile.read(dataset_path + filename)
        wav_data = wav_readonly_data / 32767

        # left_max_amplitude = max(map(abs, wav_data[:, 0]))
        # right_max_amplitude = max(map(abs, wav_data[:, 1]))
        # ILD = right_max_amplitude - left_max_amplitude

        left_first_occurrence = next(
            (i for i, v in enumerate(wav_data[:, 0]) if v), None
        )
        right_first_occurrence = next(
            (i for i, v in enumerate(wav_data[:, 1]) if v), None
        )

        real_left_wav_data = wav_data[left_first_occurrence:, 0]
        real_right_wav_data = wav_data[right_first_occurrence:, 1]

        left_amplitude_square_sum = np.sum(real_left_wav_data**2)
        right_amplitude_square_sum = np.sum(real_right_wav_data**2)

        ILD = 10 * np.log10(right_amplitude_square_sum / left_amplitude_square_sum)
        # ILD = ILD_from_wav(wav_data)
        ILD_list.append(ILD)

    # print(f"ILD {ILD_list[:5]}")

    return ILD_list


def get_ITD(dataset_path, dataset):
    """ITD: right channel first non-zero occurrence - left channel first non-zero occurrence

    Usage:
        ITD_list = get_ITD(dataset_path, dataset)

    Args:
        dataset_path (string): path to dataset
        dataset (array of dict): dataset

    Returns:
        list of float: ITD List of dataset
    """

    print("get ITD")

    """ temporary commented

    ITD_list = []

    # get first non-zero occurrence
    for data in dataset:
        filename = data['filename']
        sr, wav_data = wavfile.read(dataset_path + filename)

        left_first_occurrence = next((i for i, v in enumerate(wav_data[:, 0]) if v), None)
        right_first_occurrence = next((i for i, v in enumerate(wav_data[:, 1]) if v), None)
        ITD = right_first_occurrence - left_first_occurrence 
        ITD_list.append(ITD)

    # print(f"ITD {ITD_list[:5]}")

    return ITD_list

    """

    return gcc_phat(dataset_path, dataset)


if __name__ == "__main__":
    # get_ITD()
    # get_ILD()

    pass
