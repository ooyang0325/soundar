
from pydub import AudioSegment
from scipy.io import wavfile
from frange import *
import numpy as np
from const_value import frequency

def generate_silence():
    """ Generate silence wav file padding 100 ms silence at the beginning of the file.

    (useless now)

    Usage:
        generate_silence()

    """
    # https://stackoverflow.com/questions/46757852/adding-silent-frame-to-wav-file-using-python

    for f in frequency:
        filename = f'../data/sin_{f}.wav'
        segment = AudioSegment.silent(duration=100) # add 100 ms of silence
        wave = AudioSegment.from_wav(filename)
        wave = segment + wave
        wave.export(f'sin_{f}_delay.wav', format='wav')

def get_ILD(dataset_path, dataset):
    """ ILD: Interaural Level Difference

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

    for data in dataset:
        filename = data['filename']
        sr, wav_readonly_data = wavfile.read(dataset_path + filename)
        wav_data = wav_readonly_data / 32767

        # left_max_amplitude = max(map(abs, wav_data[:, 0]))
        # right_max_amplitude = max(map(abs, wav_data[:, 1]))
        # ILD = right_max_amplitude - left_max_amplitude

        left_amplitude_square_sum = np.sum(wav_data[:, 0] ** 2)
        right_amplitude_square_sum = np.sum(wav_data[:, 1] ** 2)

        ILD = 10 * np.log10(right_amplitude_square_sum / left_amplitude_square_sum)
        # ILD = ILD_from_wav(wav_data)
        ILD_list.append(ILD)

    # print(f"ILD {ILD_list[:5]}")

    return ILD_list

def get_ITD(dataset_path, dataset):
    """ ITD: right channel first non-zero occurrence - left channel first non-zero occurrence

    Usage:
        ITD_list = get_ITD(dataset_path, dataset)

    Args:
        dataset_path (string): path to dataset 
        dataset (array of dict): dataset

    Returns:
        list of float: ITD List of dataset 
    """

    print("get ITD")

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

if __name__ == '__main__':
    # get_ITD()
    # get_ILD()
    pass

