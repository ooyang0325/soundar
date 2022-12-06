import numpy as np
from scipy.io import wavfile
from frange import frange

file_path = '../../data/output/'
output_path = '../../data/mel_spec/'

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

	data_count = 3000
	data_shape = wavfile.read(file_path + 'output_0.wav')[1].shape
	dataset = np.empty(shape=(data_count, data_shape[0], 2))

	for i in frange(data_count):
		file_name = file_path + 'output_' + str(i) + '.wav'
		sr, data = wavfile.read(file_name)
		# data is a np array with 2 channels, 
		dataset[i] = data
	
	np.save('audio_sample.npy', dataset)

generate_dataset()

