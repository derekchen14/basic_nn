import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

# Coding environment parameters
print_loss = True
plot_results = False
np.random.seed(0)
X_train, y_train = sklearn.datasets.make_moons(200, noise=0.20)
num_examples = len(X_train) # training set size

# Neural network parameters
num_epochs = 1001  # number of epochs to train the set
learning_rate = 0.01 # learning rate for gradient descent
regularization = 0.01 # regularization strength (lambda)
nn_hidden_units = 3 # number of hidden units per layer
nn_layers = 3 # number of layers in the neural network
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
# Cannot change learning method (annealing, Momentum, SGD)
# Cannot change type of activation (tanH, Rectified LU, Sigmoid)
# Cannot change implement gradient checking

def feed_forward(model, inputs):
  model_weights = model['weights']
  model_biases = model['biases']
  outputs = []

  for idx, weight in enumerate(model_weights):
    logit = inputs.dot(weight) + model_biases[idx]
    if idx == (nn_layers - 1):
      exp_scores = np.exp(logit)
      final_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
      outputs.append(final_output)
    else:
      layer_output = np.tanh(logit)
      outputs.append(layer_output)
      inputs = layer_output
  return outputs

def calculate_loss(model):
  outputs = feed_forward(model, X_train)
  out = outputs[nn_layers-1]
  predicted_output = out[range(num_examples), y_train]
  loss = -np.sum(np.log(predicted_output))
  return loss

def plot_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    outputs = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(outputs[nn_layers-1], axis=1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PiYG)

def neural_network(nn_hidden_units, alpha, num_epochs=2000):
  np.random.seed(0)  # Initialize all the parameters to random values
  model_weights = []
  model_biases = []
  for layer in xrange(1, nn_layers+1):
  # for layer in xrange(nn_layers, 0, -1):
    if layer == 1:
      weight = np.random.randn(nn_input_dim, nn_hidden_units) / np.sqrt(nn_input_dim)  # (2x3)
      bias = np.zeros((1, nn_hidden_units))  # (1x3)
    elif layer == nn_layers:
      weight = np.random.randn(nn_hidden_units, nn_output_dim) / np.sqrt(nn_hidden_units)  #(3x2)
      bias = np.zeros((1, nn_output_dim))  # (1x2)
    else:
      weight = np.random.randn(nn_hidden_units, nn_hidden_units) / np.sqrt(nn_hidden_units)  #(3x3)
      bias = np.zeros((1, nn_hidden_units))  # (1x3)
    model_weights.append(weight)
    model_biases.append(bias)
  model = { 'weights': model_weights, 'biases': model_biases }

  # Gradient descent. For each batch...
  for i in xrange(0, num_epochs):
    outputs = feed_forward(model, X_train)
    outputs.insert(0, X_train)
    updated_weights = []
    updated_biases = []
    delta = np.array([])

    for layer in xrange(nn_layers, 0, -1):
      mod = layer - 1
      # Requires a different formula since gradient of softmax
      if layer == nn_layers:
        delta = outputs[layer]
        delta[range(num_examples), y_train] -= 1
        dW = (outputs[mod].T).dot(delta)
        db = np.sum(delta, axis=0, keepdims=True)
      # Whereas this is the gradient of the tanh layer
      else:
        future_error = delta.dot(model['weights'][layer].T)
        partial_derivative = 1 - np.square(outputs[layer])
        delta = future_error * partial_derivative      # (200x3)
        dW = (outputs[mod].T).dot(delta)
        db = np.sum(delta, axis=0, keepdims=True)
      dW += regularization * model['weights'][mod]
      updated_weights.insert(0, model['weights'][mod] - (learning_rate * dW))
      updated_biases.insert(0, model['biases'][mod] - (learning_rate * db))

    model['weights'] = updated_weights
    model['biases'] = updated_biases

    if print_loss and i % 100 == 0:
      data_loss = calculate_loss(model)/num_examples
      print "Loss after iteration %i: %.4f" % (i, data_loss)

  return model

# Build a model with a 3-dimensional hidden layer
model = neural_network(nn_hidden_units, learning_rate, num_epochs)

# Plot the decision boundary
if plot_results:
  formula = lambda x: feed_forward(model, x)
  plot_boundary(formula, X_train, y_train)
  plt.title("Decision Boundary for hidden layer with 3 units")
  plt.show()