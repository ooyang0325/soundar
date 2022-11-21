import os

file_path = '../data/output/'

for i in range(10):
	file_name = file_path + 'output_' + '0' * 4 + str(i) + '.wav'
	# os.rename(file_name, 'output_' + str(i))
	os.rename('output_' + str(i), 'output_' + str(i) + '.wav')

