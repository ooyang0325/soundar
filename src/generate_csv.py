import json
import pandas as pd
from math import atan2, pi, floor, sqrt, ceil

file_path = '../data/'

def build_coordinate_mapping_table(count=(8, 8)):
	"""
	mapping coordinate to block_id, coordinates have to be multiple of 50

	count refers to the coordinate table size
	now only support 8 x 8 and 4 x 4

	table and dic seem to be useless, but I'd like to pretend nothing happened :/
	"""

	nx, ny = 0, 0
	table = [[0 for _ in range(8)] for _ in range(8)]
	dx = [1, -1, -1, 1]
	dy = [1, 1, -1, -1]
	dic = {}
	invdic = {}
	for i in range(4):
		for j in range(i + 1):
			tx = i * i + j
			ty = i * i + 2 * i - j
			for d in range(4):
				if d % 2 == 0:
					dic[tx + 16 * d] = (50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))
					dic[ty + 16 * d] = (50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))
					invdic[(50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))] = tx + 16 * d
					invdic[(50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))] = ty + 16 * d
				if d % 2 == 1:
					dic[ty + 16 * d] = (50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))
					dic[tx + 16 * d] = (50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))
					invdic[(50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))] = ty + 16 * d
					invdic[(50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))] = tx + 16 * d

			# first quadrant
			table[3 - j][4 + i] = tx
			table[3 - i][4 + j] = ty
			# second quadrant
			table[3 - i][3 - j] = tx + 16
			table[3 - j][3 - i] = ty + 16
			# third quadrant
			table[4 + j][3 - i] = tx + 32
			table[4 + i][3 - j] = ty + 32
			# fourth quadrant
			table[4 + i][4 + j] = tx + 48
			table[4 + j][4 + i] = ty + 48

	

	# print(*table, sep='\n')

	dic = dict(sorted(dic.items()))
	invdic = dict(sorted(invdic.items(), key=lambda x: x[1]))

	if count == (4, 4):
		tmp = []
		for tup, v in invdic.items():
			if v % 16 == 2:
				invdic[tup] = v // 16 * 4
			elif v % 16 == 10:
				invdic[tup] = v // 16 * 4 + 1
			elif v % 16 == 12:
				invdic[tup] = v // 16 * 4 + 2
			elif v % 16 == 14:
				invdic[tup] = v // 16 * 4 + 3
			else:
				tmp.append(tup)
		for v in tmp:
			del invdic[v]


	return invdic


def split_json():
	"""
	Fix problem of the original json data.
	(It's likely to be useless now)
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

def add_coordinate(count):
	"""
	Add coordinate label to the json file.
	Coordinate is labeled like this: https://i.imgur.com/upXuv8t.png

	count refers to the coordinate table size
	now only support 8 x 8 and 4 x 4
	"""

	coord_table = build_coordinate_mapping_table((4, 4))
	if count == (4, 4):
		block_size = 100
		block_size
	else:
		block_size = 50 

	with open(file_path + 'new_output.json') as f:
		dic = json.load(f)
		dic['block'] = []
		for x, y, filename in zip(dic['x'], dic['y'], dic['filename']):
			nx = (abs(x) // block_size + (1 if abs(x) % block_size != 0 else 0)) * (1 if x > 0 else -1) * block_size
			ny = (abs(y) // block_size + (1 if abs(y) % block_size != 0 else 0)) * (1 if y > 0 else -1) * block_size
			if nx == 0: nx = block_size
			if ny == 0: ny = block_size
			# print(x, y, nx, ny)
			block = coord_table[(nx, ny)]
			dic['block'].append(block)

		with open('../data/output_with_coordinate.json', 'w') as output:
			json.dump(dic, output, indent=4)


def json_to_csv():
	"""
	Turn json file to csv file.
	"""
	with open('../data/output_with_coordinate.json') as f:
		json_file = json.load(f)
		df = pd.read_json('../data/output_with_coordinate.json')
		df.to_csv(file_path + 'dataset.csv', index=False)

if __name__ == '__main__':
	exit()
	add_coordinate()
	json_to_csv()


