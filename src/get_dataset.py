import pandas as pd
import numpy as np
from PIL import Image

file_path = '../data/'

df = pd.read_csv(file_path + 'dataset.csv')

cnt = [0 for _ in range(64)]

for i in range(3000):
	
	cnt[df['block'][i]] += 1

print(cnt)

