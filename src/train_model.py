
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_blobs

from get_data import get_train_data, train_test_split
import const_value

def train_kmeans_model(freq):
    """ Train model.

    """

    train_data, true_label = get_train_data(freq=const_value.frequency[0],data_count=500)
    # print(train_data.shape, true_label.shape)
    # print(train_data[:5], true_label[:5])

    (x, true_labels), (valid_x, valid_label) = train_test_split(train_data, true_label)
    # print(x.shape, true_labels.shape, valid_x.shape, valid_label.shape)

    kmeans = KMeans(n_clusters=13).fit(x)
    labels = kmeans.predict(x)

    plt.figure(1)
    plt.scatter(x[:, 0], x[:, 1], c=true_labels/15, s=30, cmap='viridis')

    plt.figure(2)
    plt.scatter(x[:, 0], x[:, 1], c=labels/15, s=30, cmap='viridis')

    plt.show()

    pass

def train_gmm_model(freq=const_value.frequency[0]):
    """ Train model.

    """

    train_data, true_label = get_train_data(freq=freq,data_count=500)
    # print(train_data.shape, true_label.shape)
    # print(train_data[:5], true_label[:5])

    (x, true_labels), (valid_x, valid_label) = train_test_split(train_data, true_label)
    # print(x.shape, true_labels.shape, valid_x.shape, valid_label.shape)

    gmm = GaussianMixture(n_components=13, covariance_type='tied').fit(x)
    labels = gmm.predict(x)

    plt.figure(1)
    plt.scatter(x[:, 0], x[:, 1], c=true_labels/15, s=30, cmap='viridis')

    plt.figure(2)
    plt.scatter(x[:, 0], x[:, 1], c=labels/15, s=30, cmap='viridis')

    plt.show()


if __name__ == '__main__':
    train_gmm_model(freq=const_value.frequency[2])

    # train_kmeans_model(freq=const_value.frequency[0])
