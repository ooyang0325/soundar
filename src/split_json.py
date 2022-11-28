import json

file_path = '../data/'

def split_json():

	f = open(file_path + 'output_1.json')

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

	output = open('../data/new_output.json', 'w')
	json.dump(dic, output, indent=4)

	f.close()
	output.close()





