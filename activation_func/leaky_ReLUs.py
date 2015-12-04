import data
import numpy as np
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt

# Coding environment parameters
print_loss = True
plot_results = False
gradient_checking = False
annealing = False  # Set initial learning rate to 0.02
# Data Gathering and pre-processing
X_train, y_train = data.generate(0, 200)
num_examples = len(X_train) # training set size

# Neural network parameters
num_epochs = 1401  # number of epochs to train the set
learning_rate = [0.01, 900] # learning rate for gradient descent
regularization = 0.01 # regularization strength (lambda)
nn_hidden_units = 3 # number of hidden units per layer
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
activation_type = "relu" # or "sigm" or "rect"

def feed_forward(model, inputs, activation_function=0):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  if activation_function == 0:
    activation_function = set_nonlinearity("tanh", 1)
  z1 = inputs.dot(W1) + b1
  a1 = activation_function(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2)
  a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  return a2, a1

def calculate_loss(model):
  out, a1 = feed_forward(model, X_train)
  predicted_output = out[range(num_examples), y_train]
  loss = -np.sum(np.log(predicted_output))
  # regularization data loss
  # sum_first_weights = np.sum(np.square(model['W1']))
  # sum_second_weights = np.sum(np.square(model['W2']))
  # loss += regularization/2 * (sum_first_weights + sum_second_weights)
  return loss

def check_gradient(model, actual_gradient, epsilon=0.0001):
  est_gradients = np.array([])
  for left in np.arange(nn_hidden_units):
    for right in np.arange(nn_output_dim):
      weight_plus = np.copy(model['W2'])
      weight_minus = np.copy(model['W2'])
      weight_plus[left,right] += epsilon
      weight_minus[left,right] -= epsilon
      model_plus = { 'W1': model['W1'], 'b1': model['b1'],'W2': weight_plus, 'b2': model['b2']}
      model_minus = {'W1': model['W1'], 'b1': model['b1'],'W2': weight_minus, 'b2': model['b2']}

      gradplus = calculate_loss(model_plus)
      gradminus = calculate_loss(model_minus)
      estimated_gradient = (gradplus - gradminus)/(2*epsilon)
      est_gradients = np.append(est_gradients, estimated_gradient)

  ratio = est_gradients/actual_gradient.ravel()
  print np.round(ratio,4)

# 1st order gives cost function, 2nd order gives gradient, 3rd order gives hessian
def set_nonlinearity(fname, order):
  tanh = {1: (lambda x: np.tanh(x)), 2:(lambda x: (1 - np.square(x)) )}
  sigm = {1: (lambda x: 1/(1 + np.exp(-x))), 2:(lambda y: y*(1-y) )}
  relu = {1: (lambda y: activate_relu(y)), 2:(lambda y: derive_relu(y))}
  leak = {1: (lambda y: activate_leaky(y)), 2:(lambda y: derive_leaky(y))}
  maxo = {1: (lambda y: np.tanh(x)), 2:(lambda y: (1 - np.square(x)) )}
  if fname == "tanh":
    return tanh[order]
  elif fname == "sigm":
    return sigm[order]
  elif fname == "relu":
    return relu[order]
  elif fname == "leak":
    return leak[order]
  elif fname == "maxo":
    return maxo[order]

def activate_relu(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = max(0,x)
  return matrix
def derive_relu(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = 0 if (x<0) else 1
  return matrix
# paremetric ReLU simply makes 0.01 a variable parameter
def activate_leaky(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = 0.01 * x if (x<0) else x
  return matrix
def derive_leaky(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = 0.01 if (x<0) else 1
  return matrix


def neural_network(nn_hidden_units, alpha, num_epochs=2000):
  np.random.seed(0)  # Initialize all the parameters to random values
  W1 = np.random.randn(nn_input_dim, nn_hidden_units) / np.sqrt(nn_input_dim)  # (2x3)
  b1 = np.zeros((1, nn_hidden_units))  # (1x3)
  W2 = np.random.randn(nn_hidden_units, nn_output_dim) / np.sqrt(nn_hidden_units)  #(3x2)
  b2 = np.zeros((1, nn_output_dim))
  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }
  activation_function = set_nonlinearity(activation_type, 1)
  gradient_function = set_nonlinearity(activation_type, 2)

  # Gradient descent. For each batch...
  for i in xrange(0, num_epochs):
    # generate outputs
    a2, a1 = feed_forward(model, X_train, activation_function)
    # back progpagation
    delta2 = a2
    delta2[range(num_examples), y_train] -= 1
    dW2 = (a1.T).dot(delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    future_error = delta2.dot(W2.T)
    partial_derivative = gradient_function(a1)
    delta1 = future_error * partial_derivative      # (200x3)
    dW1 = (X_train.T).dot(delta1)                   # (2x3)
    db1 = np.sum(delta1, axis=0)                    # (1x3)

    # Add regularization terms (b1 & b2 don't have regularization terms)
    dW2 += regularization * W2
    dW1 += regularization * W1

    # Parameter update of the weights and biases
    learning_rate = (alpha[0]*alpha[1])/(alpha[1]+i) if annealing else alpha[0]
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Assign new parameters to the model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

    if gradient_checking and i % 200 == 0:
      print "Gradient Check after round %i:" % i
      actual_gradient = dW2 - (regularization * W2)
      check_gradient(model, actual_gradient)
    if print_loss and i % 100 == 0:
      data_loss = calculate_loss(model)/num_examples
      print "Loss after iteration %i: %.4f" % (i, data_loss)

  return model

# Build a model with a 3-dimensional hidden layer
model = neural_network(nn_hidden_units, learning_rate, num_epochs)

# Plot the decision boundary
if plot_results:
  formula = lambda x: feed_forward(model, x)
  data.plot_boundary(formula, X_train, y_train)
  plt.title("Decision Boundary for hidden layer with 3 units")
  plt.show()