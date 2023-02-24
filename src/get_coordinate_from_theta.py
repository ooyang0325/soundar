import numpy as np
from math import cos, sqrt
from scipy.io import wavfile
from get_data import get_random_data
import matplotlib.pyplot as plt
import binaural_cues

import const_value

dataset_path = '../dataset/'

def get_const_value_d_and_v(freq):
    """ Calculate d and v by following formula
    d: distance between two microphones
    v: velocity of sound


    """

    dataset = get_random_data(freq, data_count=3900)
    k = get_ITD_on_x_axis(freq, dataset)

    dlist = []

    for data in dataset:
        if data['theta'] != 15:
            continue
        filename = data['filename']
        sr, wav_readonly_data = wavfile.read(dataset_path + f'{freq}_delay/'+ filename)
        wav_data = wav_readonly_data / 32767

        total_len = len(wav_data[:, 0]) + len(wav_data[:, 1])
        left_fft = np.fft.rfft(wav_data[:, 0], n=total_len)
        right_fft = np.fft.rfft(wav_data[:, 1], n=total_len)
        r = left_fft * np.conj(right_fft)

        cross_correlation = np.fft.irfft(r / np.abs(r), n=(total_len * 16))
        max_shift = 16 * total_len // 2
        cross_correlation = np.concatenate((cross_correlation[-max_shift:], cross_correlation[:max_shift]))
        shift = np.argmax(cross_correlation) - max_shift

        ITD_ab = shift / float(sr * 16)

        # print(ITD_ab, k)
        k2 = 2 * abs(ITD_ab) / k
        R = data['r']
        theta = data['theta']
        # print(R, theta)
        # print(f'k2 cos', k2, theta, cos(theta), k2 ** 2 - 4 * cos(theta) ** 2, k2 ** 2 - 4)

        try:
            d = 2 * R / k2 * sqrt((k2 ** 2 - 4 * (cos(theta) ** 2)) / (k2 ** 2 - 4))
        except:
            print('qaq')
            continue
        dlist.append(d)

    dlist = np.array(dlist)
    print(dlist[:100])
    d = np.average(dlist)

    # coordinate x = 10, y = 0

    return d

def check_ITD(freq):
    ITD_list = []
    rlist = []
    all_dataset = get_random_data(freq, data_count=3900, rand=False)
    dataset = [data for data in all_dataset if data['theta'] == 30]
    ITD_list = binaural_cues.get_ITD(dataset_path + f'{freq}_delay/', dataset)
    rlist = [data['r'] for data in dataset]

    print(rlist)

    plt.plot(rlist, ITD_list)
    plt.show()
    
    pass

def get_ITD_on_x_axis(freq, dataset):
    """
    get ITD on x axis
    """
    print("get ITD")

    ITD_list = []

    # get first non-zero occurrence
    for data in dataset:
        if data['y'] != 0:
            continue
        filename = data['filename']
        sr, wav_data = wavfile.read(dataset_path + f'{freq}_delay/' + filename)

        left_first_occurrence = next((i for i, v in enumerate(wav_data[:, 0]) if v), None)
        right_first_occurrence = next((i for i, v in enumerate(wav_data[:, 1]) if v), None)
        ITD = right_first_occurrence - left_first_occurrence 
        ITD_list.append(abs(ITD))
 
    ITD_list = np.array(ITD_list)
    average = np.average(ITD_list)

    print(f'avergae ITD: {average}')

    return average


if __name__ == '__main__':
    freq = const_value.frequency[1]
    # dataset = get_random_data(freq, data_count=3900)
    # k = get_ITD_on_x_axis(freq, dataset)
    # d, _ = get_const_value_d_and_v(freq)
    # print(d)
    check_ITD(freq)
