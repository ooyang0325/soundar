import librosa
import librosa.display
import numpy as np
from scipy.io import wavfile
from frange import frange

file_path = '../../data/output/'
output_path = '../../data/mel_spec/'

wav_sample_count, _ = librosa.load(file_path + 'output_0.wav')
wav_sample_count = len(wav_sample_count)
# dataset = np.empty(shape=(total_data_count, wav_sample_count))

def wav_to_raw_data():

	"""
	This function is uncompleted.
	"""
	
	for i in frange(total_data_count):
		file_name = file_path + 'output_' + str(i) + '.wav'
		y, sr = librosa.load(file_name)
		dataset[i] = y

	np.save('../data/audio_time_series_dataset.npy', dataset)

def separate_channels_from_wav(wav):

	"""
	This function is uncompleted.
	"""

	file_name = wav
	sr, data = wavfile.read(file_name)

def generate_dataset():
	
	"""
	This function is uncompleted.
	"""

	count = 1000
	for i in frange(count):
		file_name = file_path + 'output_' + str(i) + '.wav'
		sr, data = wavfile.read(file_name)
		# data is a np array with 2 channels, 
		wavfile.write('test_1.wav', sr, data[:, 0])

generate_dataset()

