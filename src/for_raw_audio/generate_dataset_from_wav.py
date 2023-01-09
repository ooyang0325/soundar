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

    data_count = 3000
    data_shape = wavfile.read(file_path + '1046.5/output_0.wav')[1].shape
    dataset = np.empty(shape=(data_count, 4410, 2)) # 4410 from 0.1 second, 2 for 2 channels

    cnt = 0

    for freq in frequency:
        for i in frange(data_count):
            file_name = file_path + str(freq) + '/output_' + str(i) + '.wav'
            sr, data = wavfile.read(file_name)
            # data is a np array with 2 channels, 

            # clip data from the first original point
            for j in range(1, len(data)):
                if data[j - 1][0] < 0 and data[j][0] > 0:
                    data = data[j:j + 4410]
                    break

            if freq > 1000 and i < 1:
                for j in data:
                    print(j)

            # data = data[4096:4096 + 4410]
            
            dataset[i] = data
    
        np.save(f'audio_sample_{str(freq)}.npy', dataset)

generate_dataset()

