import matplotlib.pyplot as plt
import const_value
from get_data import get_train_data, train_test_split

if __name__ == '__main__':
    train_data, true_label = get_train_data(freq=const_value.frequency[0], data_count=3900)
    print(train_data.shape, true_label.shape)

    plt.scatter(train_data[:, 1], train_data[:, 0], c=true_label, s=20, cmap='viridis')
    plt.xlabel("ITD")
    plt.ylabel("ILD")

    plt.show()

    pass


