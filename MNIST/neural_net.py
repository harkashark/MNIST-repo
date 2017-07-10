import numpy as np
import load_data as ld

class Network(object):

    def __init__(self, sizes): #sizes will equal (784, 10)
        self.biases = np.random.randn(sizes[1],)
        self.weights = np.random.randn(sizes[0], sizes[1])

    def feedforward(self, input_image, biases): #input_image as a member of train_images or other list
#        output_vector = [] ==== redundant
        input_dot_weights = np.dot(np.asarray(input_image), self.weights)
        x = np.add(input_dot_weights, self.biases)
        return sigmoid_f(x)

def sigmoid_f(x):
    return 1.0 / (1.0 + np.exp(-x))

net = Network((784, 10))
correct_count = 0
incorrect_count = 0
(train_images, train_labels, validation_images,
    validation_labels, test_images, test_labels) = ld.get_data()
for image, label in zip(train_images, train_labels): #image data imported from load_data.py
    output = list(net.feedforward(image, net.biases))
    mx = max(output)
    mx_index = output.index(mx)
    if mx_index == label:
        correct_count += 1
    else:
        incorrect_count +=1
    percent_correct = 100*(float(correct_count)/
                            (float(correct_count) + float(incorrect_count)))
print "Percent correct is %f%%" % percent_correct
