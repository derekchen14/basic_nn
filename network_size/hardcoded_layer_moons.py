import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

# Coding environment parameters
print_loss = True
plot_results = False
gradient_checking = False
annealing = False  # Set initial learning rate to 0.02
# Data Gathering and pre-processing
np.random.seed(0)
X_train, y_train = sklearn.datasets.make_moons(200, noise=0.20)
num_examples = len(X_train) # training set size

# Neural network parameters
num_epochs = 1001  # number of epochs to train the set
learning_rate = [0.01, 900] # learning rate for gradient descent
regularization = 0.01 # regularization strength (lambda)
nn_hidden_units = 3 # number of hidden units per layer
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
activation_type = "tanh" # or "sigm" or "rect"

def feed_forward(model, inputs, activation_function=0):
  W1, W2, W3 = model['W1'], model['W2'], model['W3']
  b1, b2, b3 = model['b1'], model['b2'], model['b3']
  if activation_function == 0:
    activation_function = set_nonlinearity("tanh", 1)
  z1 = inputs.dot(W1) + b1
  a1 = activation_function(z1)
  z2 = a1.dot(W2) + b2
  a2 = activation_function(z2)
  z3 = a2.dot(W3) + b3
  exp_scores = np.exp(z3)
  a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  return a3, a2, a1

def calculate_loss(model):
  out, a2, a1 = feed_forward(model, X_train)
  predicted_output = out[range(num_examples), y_train]
  loss = -np.sum(np.log(predicted_output))
  # regularization data loss
  # sum_first_weights = np.sum(np.square(model['W1']))
  # sum_second_weights = np.sum(np.square(model['W2']))
  # sum_third_weights = np.sum(np.square(model['W3']))
  # total_weights = sum_first_weights + sum_second_weights + sum_third_weights
  # loss += regularization/3 * (total_weights)
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
  rect = {1: (lambda y: activate_relu(y)), 2:(lambda y: derive_relu(y))}
  if fname == "tanh":
    return tanh[order]
  elif fname == "sigm":
    return sigm[order]
  elif fname == "rect":
    return rect[order]

def activate_relu(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = max(0,x)
  return matrix
def derive_relu(matrix):
  for x in np.nditer(matrix, op_flags=["readwrite"]):
    x[...] = 0 if (x<0) else 1
  return matrix

def plot_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    a3, a2, a1 = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(a3, axis=1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PiYG)

def neural_network(nn_hidden_units, alpha, num_epochs=2000):
  np.random.seed(0)  # Initialize all the parameters to random values
  W1 = np.random.randn(nn_input_dim, nn_hidden_units) / np.sqrt(nn_input_dim)  # (2x3)
  b1 = np.zeros((1, nn_hidden_units))  # (1x3)
  W2 = np.random.randn(nn_hidden_units, nn_hidden_units) / np.sqrt(nn_hidden_units)  #(3x3)
  b2 = np.zeros((1, nn_hidden_units))  # (1x3)
  W3 = np.random.randn(nn_hidden_units, nn_output_dim) / np.sqrt(nn_hidden_units)  #(3x2)
  b3 = np.zeros((1, nn_output_dim))  # (1x2)
  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }
  activation_function = set_nonlinearity(activation_type, 1)
  gradient_function = set_nonlinearity(activation_type, 2)

  # Gradient descent. For each batch...
  for i in xrange(0, num_epochs):
    # generate outputs
    a3, a2, a1 = feed_forward(model, X_train, activation_function)

    # back progpagation
    delta3 = a3
    delta3[range(num_examples), y_train] -= 1
    dW3 = (a2.T).dot(delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)

    future_error2 = delta3.dot(W3.T)
    partial_derivative2 = gradient_function(a2)
    delta2 = future_error2 * partial_derivative2      # (200x3)
    dW2 = (a1.T).dot(delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    future_error1 = delta2.dot(W2.T)
    partial_derivative1 = gradient_function(a1)
    delta1 = future_error1 * partial_derivative1      # (200x3)
    dW1 = (X_train.T).dot(delta1)                   # (2x3)
    db1 = np.sum(delta1, axis=0)                    # (1x3)

    # Add regularization terms (b1 & b2 don't have regularization terms)
    dW3 += regularization * W3
    dW2 += regularization * W2
    dW1 += regularization * W1

    # Parameter update of the weights and biases
    learning_rate = (alpha[0]*alpha[1])/(alpha[1]+i) if annealing else alpha[0]
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Assign new parameters to the model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }

    if gradient_checking and i % 200 == 0:
      print "Gradient Check after round %i:" % i
      actual_gradient = dW3 - (regularization * W3)
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
  plot_boundary(formula, X_train, y_train)
  plt.title("Decision Boundary for hidden layer with 3 units")
  plt.show()