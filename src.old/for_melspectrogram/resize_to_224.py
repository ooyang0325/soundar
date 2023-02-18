from PIL import Image
from frange import frange

def resize_to_224():
	"""
	Resize the mel_spec image to (224, 224, 3)
	"""
	for i in frange(10000):
		file_path = f'../data/mel_spec/{i}.png'
		im = Image.open(file_path)
		im = im.resize((224, 224))
		im = im.convert("RGB")
		im.save(file_path)

resize_to_224()

