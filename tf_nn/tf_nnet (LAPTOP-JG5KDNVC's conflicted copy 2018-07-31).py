import tensorflow as tf
import numpy as np

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

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    W1 = tf.get_variable("W1", [24, 10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [24, 1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [12, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    W4 = tf.get_variable("W4", [1, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}

    return parameters

def forward_propagation(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    #transpose input matrix so columns of weights = rows of input to use matrix mult.

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    with tf.device('/gpu:0'):                                  # Numpy Equivalents:
        Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
        A1_dropout = tf.nn.dropout(A1, keep_prob)              # apply dropout (*)

        Z2 = tf.add(tf.matmul(W2, A1_dropout), b2)             # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
        A2_dropout = tf.nn.dropout(A2, keep_prob)              # apply dropout (*)

        Z3 = tf.add(tf.matmul(W3, A2_dropout), b3)                     # Z3 = np.dot(W3,Z2) + b3
        A3 = tf.nn.relu(Z3)
        A3_dropout = tf.nn.dropout(A3, keep_prob)              # apply dropout (*)

        Z4 = tf.add(tf.matmul(W4, A3_dropout), b4)

    return Z4

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


def predictSurvivors(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4}

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])
    keep_prob = tf.placeholder(tf.float32)

    z4 = forward_propagation_for_predict(x, params, keep_prob)
    p = tf.sigmoid(z4)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X, keep_prob: 0.99})

    return prediction

def forward_propagation_for_predict(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    A1_dropout = tf.nn.dropout(A1, keep_prob)              # apply dropout (*)

    Z2 = tf.add(tf.matmul(W2, A1_dropout), b2)             # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    A2_dropout = tf.nn.dropout(A2, keep_prob)              # apply dropout (*)

    Z3 = tf.add(tf.matmul(W3, A2_dropout), b3)                     # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    A3_dropout = tf.nn.dropout(A3, keep_prob)              # apply dropout (*)

    Z4 = tf.add(tf.matmul(W4, A3_dropout), b4)

    return Z4
