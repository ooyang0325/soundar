import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from frange import frange

file_path = '../data/output/'
output_path = '../data/mel_spec/'

fig = plt.figure(figsize=[0.72, 0.72])
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)

total_data_count = 10000

wav_sample_count, _ = librosa.load(file_path + 'output_0.wav')
wav_sample_count = len(wav_sample_count)
dataset = np.empty(shape=(total_data_count, wav_sample_count))

def wav_to_melspectrogram():

	for i in frange(total_data_count):
		file_name = file_path + 'output_' + str(i) + '.wav'
		y, sr = librosa.load(file_name)

		S = librosa.feature.melspectrogram(y=y, sr=sr)
		librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
		plt.savefig(output_path + str(i) + '.png', dpi=800, bbox_inches='tight', pad_inches=0)
		if i % 100 == 0:
			print('fubuki<3', i)

def wav_to_raw_data():
	
	for i in frange(total_data_count):
		file_name = file_path + 'output_' + str(i) + '.wav'
		y, sr = librosa.load(file_name)
		dataset[i] = y

	np.save('../data/audio_time_series_dataset.npy', dataset)

wav_to_raw_data()

