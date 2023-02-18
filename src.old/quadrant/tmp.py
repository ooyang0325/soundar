import librosa 
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

freq = 1046.5
file_path = '../../data/dataset/'
i = 0
file_name = file_path + str(freq) + '/output_' + str(i) + '.wav'
sr, y = wavfile.read(file_name)
print(y[:5])
y, sr = librosa.load(file_name, sr=44100, mono=False)
print(y[:5])

exit(0)

S = librosa.feature.melspectrogram(y=y[0], sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
plt.savefig(mel_file_name, bbox_inches='tight', pad_inches=0)

