import os

file_path = '../data/output/'

for i in range(1000, 10000):
	file_name = file_path + 'output_' + '0' * 1 + str(i) + '.wav'
	os.rename(file_name, file_path + 'output_' + str(i) + '.wav')

