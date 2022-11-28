import json
import pandas as pd
from math import atan2, pi, floor

file_path = '../data/'

def split_json():
	"""
	Fix problem of the original json data.
	"""
	with open(file_path + 'output_1.json') as f:
		data = f.read().replace('}', '},')
		data = data[:-2]
		data = '[' + data + ']'
		lst = eval(data)

		dic = {'x': [], 'y': [], 'filename': []}

		for d in lst:
			for v in d['x']:
				dic['x'].append(v)
			for v in d['y']:
				dic['y'].append(v)
			for v in d['filename']:
				new_id = int(v[7:12])
				dic['filename'].append('output_' + str(new_id) + '.wav')

		with open('../data/new_output.json', 'w') as output:
			json.dump(dic, output, indent=4)

def add_coordinate():
	"""
	Add coordinate label to the json file.
	"""
	global file_path
	with open(file_path + 'new_output.json') as f:
		dic = json.load(f)
		dic['distance'] = []
		dic['quadrant'] = []
		# quad = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
		for x, y, filename in zip(dic['x'], dic['y'], dic['filename']):
			dist = (x * x + y * y) // 625 + 1
			dic['distance'].append(dist)
			quadrant = 0
			degree = atan2(y, x) + (2 * pi if y < 0 else 0)
			quadrant = floor(degree / (pi / 4)) + 1
			dic['quadrant'].append(quadrant)

		with open('../data/output_with_coordinate.json', 'w') as output:
			json.dump(dic, output, indent=4)


def json_to_csv():
	"""
	Turn json file to csv file.
	"""
	global file_path
	with open('../data/output_with_coordinate.json') as f:
		json_file = json.load(f)
		df = pd.read_json('../data/output_with_coordinate.json')
		df.to_csv(file_path + 'dataset.csv')

add_coordinate()
json_to_csv()


