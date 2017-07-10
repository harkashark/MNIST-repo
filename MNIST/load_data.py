import urllib
import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_data():
    #url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
    #urllib.urlretrieve(url, "mnist.pkl.gz")

    with gzip.open('mnist.pkl.gz', 'rb') as d:
        mnist_data = pickle.load(d)

    train_images, train_labels = mnist_data[0][0], mnist_data[0][1]
    validation_images, validation_labels = mnist_data[1][0], mnist_data[1][1]
    test_images, test_labels = mnist_data[2][0], mnist_data[2][1]
    return (train_images, train_labels, validation_images,
    validation_labels, test_images, test_labels)

def show_data_dimensions(train, validation,test):
    print "Number of images in train set: %d" % len(train)
    print "Number of images in validation set: %d" % len(validation)
    print "Number of images in test set: %d" % len(test)

def char_image(data):
    plt.imshow(data.reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()

def get_trial_data():
    with gzip.open('mnist.pkl.gz', 'rb') as d:
        mnist_data = pickle.load(d)
    trial_images, trial_labels = mnist_data[0][0][0:3000], mnist_data[0][1][0:3000]
    return trial_images, trial_labels
