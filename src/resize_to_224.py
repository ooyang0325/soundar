from PIL import Image
from tqdm import tqdm

for i in tqdm(range(10000)):
	file_path = f'../data/mel_spec/{i}.png'
	im = Image.open(file_path)
	im = im.resize((224, 224))
	im.save(file_path)



