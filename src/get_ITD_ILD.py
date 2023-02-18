
from scipy.io import wavfile
from frange import *

file_path = '../data/dataset/'
frequency = [1046.5, 261.63, 130.81]

def get_ILD():
    filename = 'test_soundwave.wav'
    data = wavfile.read(filename)
    print(data)

def get_ITD():
    pass

if __name__ == 