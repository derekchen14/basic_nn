import data
import numpy as np
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt

# Coding environment parameters
print_loss = True
plot_results = False
gradient_checking = False
# Data Gathering and pre-processing
X_train, y_train = data.generate(0, 200)
num_examples = len(X_train) # training set size

# Neural network parameters
num_epochs = 1401  # number of epochs to train the set
learning_rate = 0.0001 # learning rate for gradient descent
regularization = 0.01 # regularization strength (lambda)
nn_hidden_units = 3 # number of hidden units per layer
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
batch_size = 10 # examples in each mini-batch, should be around 256

def feed_forward(model, X):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  z1 = X.dot(W1) + b1
  a1 = np.tanh(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2)
  a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  return a2, a1

def calculate_loss(model):
  out, a1 = feed_forward(model, X_train)
  predicted_output = out[range(num_examples), y_train]
  loss = -np.sum(np.log(predicted_output))
  return loss

def neural_network(nn_hidden_units, print_loss, num_passes=2000):
  np.random.seed(0)  # Initialize all the parameters to random values
  W1 = np.random.randn(nn_input_dim, nn_hidden_units) / np.sqrt(nn_input_dim)  # (2x3)
  b1 = np.zeros((1, nn_hidden_units))  # (1x3)
  W2 = np.random.randn(nn_hidden_units, nn_output_dim) / np.sqrt(nn_hidden_units)  #(3x2)
  b2 = np.zeros((1, nn_output_dim))
  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

  # Gradient descent. For each batch...
  for i in xrange(0, num_epochs):
    np.random.shuffle(X_train)
    for j in xrange(0,num_examples/batch_size):
      start_ex = j*batch_size
      end_ex = (j+1)*batch_size
      X_batch = X_train[start_ex:end_ex]
      y_batch = y_train[start_ex:end_ex]
      # generate outputs
      a2, a1 = feed_forward(model, X_batch)
      # if i < 40:
      #   print a2[0:6,0:1].T
        # print a2[batch_size-1, y_batch]
      # back progpagation
      delta3 = a2
      # print delta3
      # print delta3
      delta3[batch_size-1, y_batch] -= 1
      # print delta3
      dW2 = (a1.T).dot(delta3)
      db2 = np.sum(delta3, axis=0, keepdims=True)
      delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
      # print x.T
      dW1 = np.dot(X_batch.T, delta2)
      db1 = np.sum(delta2, axis=0)

      # Add regularization terms (b1 & b2 don't have regularization terms)
      dW2 += regularization * W2
      dW1 += regularization * W1
      # Parameter update of the weights and biases
      W2 -= learning_rate * dW2
      b2 -= learning_rate * db2
      W1 -= learning_rate * dW1
      b1 -= learning_rate * db1

      # Assign new parameters to the model
      model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

      # if print_loss and j % 25 == 0:
      #   data_loss = calculate_loss(model)/num_examples
      #   print "Loss after iteration %i: %.3f" % (i, data_loss)

    if gradient_checking and i % 200 == 0:
      print "Gradient Check after round %i:" % i
      actual_gradient = dW2 - (regularization * W2)
      check_gradient(model, actual_gradient)
    if print_loss and i % 100 == 0:
      data_loss = calculate_loss(model)/num_examples
      print "Loss after iteration %i: %.4f" % (i, data_loss)

  return model

# Build a model with a 3-dimensional hidden layer
model = neural_network(nn_hidden_units, print_loss, num_epochs)