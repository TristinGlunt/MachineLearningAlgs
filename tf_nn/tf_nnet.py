import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

tf.reset_default_graph();

""" Neural network implementation, mainly from TensorFlow_Tutorial from deeplearning.ai"""
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of input vector
    n_y -- scalar, number of outputs

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y, keep_prob

def initialize_parameters(nnet_dimensions):
    """
    Initializes parameters to build a neural network with tensorflow

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    parameters = {}
    current_dim = 0
    next_dim = 0

    for i in range(0, len(nnet_dimensions)-1):

        current_dim = nnet_dimensions[i]
        next_dim = nnet_dimensions[i+1]

        W_temp = tf.get_variable("W" + str(i+1), [next_dim, current_dim], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b_temp = tf.get_variable("b" + str(i+1), [next_dim, 1], initializer = tf.zeros_initializer())

        parameters["W" + str(i+1)] = W_temp
        parameters["b" + str(i+1)] = b_temp

    return parameters

def forward_propagation(X, parameters, keep_prob, num_of_layers):
    """
    Implements the forward propagation for the model:
    LINEAR -> RELU -> LINEAR -> RELU -> ... -> SIGMOID 

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters

    Returns:
    Z -- the output of the last LINEAR unit
    """

    W = X
    for i in range(1, num_of_layers):                    #go through each layer except the output layer
        W_next = parameters["W" + str(i)]
        b = parameters["b" + str(i)]

        Z = tf.add(tf.matmul(W_next, W), b)
        A = tf.nn.relu(Z)
        A_dropout = tf.nn.dropout(A, keep_prob)

        W = A_dropout

    Z = tf.add(tf.matmul(parameters["W" + str(i+1)], A_dropout), parameters["b" + str(i+1)])

    return Z

def compute_cost(Z4, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

def generate_random_minibatches(X, Y, minibatch_size):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    trans_X = np.transpose(X)
    trans_Y = np.transpose(Y)

    tempMatrix = np.c_[trans_X, trans_Y]
    np.random.shuffle(tempMatrix)

    shuffled_X = tempMatrix[:, :-1]
    shuffled_Y = tempMatrix[:, tempMatrix.shape[1]-1].reshape(-1,1)

    shuffled_X = np.transpose(shuffled_X)
    shuffled_Y = np.transpose(shuffled_Y)

    #print(shuffled_X.shape)
    #print(shuffled_Y.shape)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/minibatch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):
        currentMinibatch = k * minibatch_size
        mini_batch_X = []
        mini_batch_Y = []
        for i in range(0, minibatch_size):
            mini_batch_X.append(shuffled_X[:, currentMinibatch+i])
            mini_batch_Y.append(shuffled_Y[:, currentMinibatch+i])
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)


    # Handling the end case (last mini-batch < mini_batch_size)
    if m % minibatch_size != 0:
        lastMinibatch = num_complete_minibatches * minibatch_size
        mini_batch_X = []
        mini_batch_Y = []
        for i in range(0, X.shape[1]-lastMinibatch):
            mini_batch_X.append(shuffled_X[:, lastMinibatch + i])
            mini_batch_Y.append(shuffled_Y[:, lastMinibatch + i])

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train, X_test, Y_test, nnet_dims, learning_rate = 0.001, num_epochs = 1000,
          minibatch_size = 712, print_cost = True):

    print("Learning rate: " + str(learning_rate) + " minibatch size: " + str(minibatch_size))
    print("Num of epochs: " + str(num_epochs))

    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    num_of_layers = len(nnet_dims) - 1

    # Create Placeholders of shape (n_x, n_y)
    X, Y, keep_prob = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(nnet_dims)

    for i in range(1, num_of_layers):
        print("W" + str(i) + " shape: " + str(parameters["W" + str(i)].shape))

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z4 = forward_propagation(X, parameters, keep_prob, num_of_layers)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z4, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = math.floor(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = generate_random_minibatches(X_train, Y_train, minibatch_size)
            counter = 0
            for minibatch in minibatches:
                counter += 1
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = np.transpose(minibatch_X)
                minibatch_Y = np.transpose(minibatch_Y)
                #print(minibatch_Y.size)

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: 0.99})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        # Calculate accuracy on the test set
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def predictSurvivors(X, parameters, num_of_layers):

    params = {}

    for i in range(1, num_of_layers):
        W_string = "W" + str(i)
        b_string = "b" + str(i)

        W = tf.convert_to_tensor(parameters[W_string])
        b = tf.convert_to_tensor(parameters[b_string])

        params[W_string] = W
        params[b_string] = b

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])
    keep_prob = tf.placeholder(tf.float32)

    z4 = forward_propagation(x, params, keep_prob, num_of_layers-1)
    p = tf.sigmoid(z4)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X, keep_prob: 0.99})

    return prediction
