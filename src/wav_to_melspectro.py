import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = '../data/output/'
output_path = '../data/mel_spec/'

fig = plt.figure(figsize=[0.72, 0.72])
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)

for i in range(7679, 10000):
	file_name = file_path + 'output_' + str(i) + '.wav'
	y, sr = librosa.load(file_name)
	S = librosa.feature.melspectrogram(y=y, sr=sr)
	librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
	plt.savefig(output_path + str(i) + '.png', dpi=800, bbox_inches='tight', pad_inches=0)
	if i % 100 == 0:
		print('fubuki<3', i)

