def neural_network(nn_hidden_units, num_epochs=2000, print_loss=False):
  np.random.seed(0)
  W1 = np.random.randn(nn_input_dim, nn_hidden_units) / np.sqrt(nn_input_dim)  # (2x3)
  b1 = np.zeros((1, nn_hidden_units))  # (1x3)
  '''
  Add a second hidden layer, making this a 3-layer neural network
  # W2 = np.random.randn(nn_hidden_units, nn_hidden_units) / np.sqrt(nn_hidden_units)
  # b2 = np.zeros((1, nn_hidden_units))
  '''
  W2 = np.random.randn(nn_hidden_units, nn_output_dim) / np.sqrt(nn_hidden_units)  #(3x2)
  b2 = np.zeros((1, nn_output_dim))

  model = {}    # This is what we return at the end

  for i in xrange(0, num_passes):

    # Feed values forward to the hidden layer
    z1 = X_train.dot(W1) + b1   # (200x3)
    # Hyperbolic Tangent activation function
    a1 = np.tanh(z1)            # (200x3)
    # Feed values forward to the output layer
    z2 = a1.dot(W2) + b2        # (200x2)
    # Softmax Classifier, generates probabilites not class estiamtes
    exp_scores = np.exp(z2)     # (200x2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # probs is our y_hat or expected output
    delta3 = probs
    # What we're trying to calculate here is the derivative of the
    # log-likelihood cost function for a softmax output layer. The
    # formula for this is actually super simple, just y_hat - y_actual
    #   where y_hat = output from the softmax function
    #   and y_actual = vector of labels
    # This is the error term of log likelihood function. It can
    #   also be interpreted as the partial derivative of the
    #   final error w.r.t. output
    # This works because delta3 is the list of all outputs, so
    #   delta3[200, y] gives the predicted output for the correct
    #   answer.  Additionally, by definition the correct answer
    #   has an output of 1, so y_actual will always be 1.
    delta3[range(num_examples), y] -= 1     # (200x2)
    # Get the error attributed to the weight using the chain rule
    # Partial derivative of the output w.r.t. the weight = a1
    # Since there is no activation function for the softmax layer
    # This is simply the gradient of the output, which is just y_hat
    dW2 = (a1.T).dot(delta3)                      # (3x2)
    #     (3x200) * (200x2)  = (3x2)
    #  3 hidden units times 2 output units
    # Sum of all error terms
    # Partial derivative of the bias w.r.t. the weight is 1
    # So no need to really calculate. If you really want, the chain
    # rule would result in:  db2 = 1 * np.sum(delta3, axis=0)
    db2 = np.sum(delta3, axis=0)                  # (1x2)

    # Error caused by the "future errors" in layers up ahead
    ft_error = delta3.dot(W2.T)
    # Partial derivative of hyperbolic tangent activation function
    pd_tanh = (1 - np.square(a1))
    # Use the chain rule to multiply together
    # This is the actual step of back-propogation where the error
    # from up ahead ia propagated to a previous layer!!!
    delta2 = ft_error * pd_tanh                   # (200x3)
    # One more time to incude the partial derivative of the logit
    # Remember, the derivative w.r.t. the weight is just the thing
    # multiplying the weight.
    # In this case that thing is just the input, X_train
    dW1 = (X_train.T).dot(delta2)                 # (2x3)
    db1 = np.sum(delta2, axis=0)                  # (1x3)

    # Add regularization terms (b1 & b2 don't have regularization terms)
    dW2 += regularization * W2
    dW1 += regularization * W1

    # Parameter update of the weights and biases
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2

    # Assign new parameters to the model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Print out the errors on each mini-batch iteration
    # This is expensive because it uses the whole dataset,
    # so we don't want to do it too often.
    if print_loss and i % 200 == 0:     # used to be 1000
      print "Loss after iteration %i: %f" %(i, calculate_loss(model))

  return model