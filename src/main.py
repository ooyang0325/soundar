import get_data
import get_ITD_ILD
import const_value

dataset_path = '../dataset/'

if __name__ == '__main__':

    freq_dataset_path = dataset_path + f'{const_value.frequency[0]}_delay/'
    print(freq_dataset_path)

    data = get_data.get_json_data(const_value.frequency[0])
    data = get_data.get_random_data(data, 50)
    print(data)
    ILD_list = get_ITD_ILD.get_ILD(freq_dataset_path, data)

    pass