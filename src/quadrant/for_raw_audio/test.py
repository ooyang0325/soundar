import numpy as np
from scipy.io import wavfile
from frange import frange

file_path = '../../data/dataset/'
output_path = '../../data/mel_spec/'

# dataset = np.empty(shape=(total_data_count, wav_sample_count))

def generate_dataset():
    
    """
    Seems to be completed?
    RT
    """

    frequency = [1046.5, 261.63, 130.81]

    data_count = 30
    data_shape = wavfile.read(file_path + '1046.5/output_0.wav')[1].shape
    print(data_shape)
    return
    dataset = np.empty(shape=(data_count * 3, data_shape[0], 2))

    cnt = 0

    for freq in frequency:
        for i in frange(data_count):
            file_name = file_path + str(freq) + '/output_' + str(i) + '.wav'
            sr, data = wavfile.read(file_name)
            # data is a np array with 2 channels, 
            dataset[cnt] = data
            cnt += 1
    
        np.save(f'audio_sample_{str(freq)}.npy', dataset)

generate_dataset()

