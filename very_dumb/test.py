import dumb_net
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

X_train = mnist.train.images.tolist()[0:1000]
Y_train = mnist.train.labels.tolist()[0:1000]

X_test = mnist.test.images.tolist()[0:1000]
Y_test = mnist.test.labels.tolist()[0:1000]

num_features = len(X_train[0])
num_labels = 10

nn = dumb_net.NN([num_features, 20, num_labels])
print(f'Training...')
nn.fit(X_train, Y_train)

print(f'Evaluating...')
accuracy = nn.evaluate(X_test, Y_test)
print(f'Accuracy: {accuracy}')
