#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Tristin Glunt / Anthony Galczak - tglunt@unm.edu - agalczak@unm.edu
# CS529 - Project 1 - Decision Trees
# This project is the first programming assignment in graduate level Machine Learning.
# It involves implementing a decision tree and its logic from scratch.
# We implemented the algorithms for entropy, gni index, information gained,
# chi square, and ID3 decision trees.
#
# Additionally, we implemented a naive version of random forests to gain
# greater accuracy. This was not required of the project.


import pandas as pd
import math
import numpy as np
import decision_tree as dt
import dt_math as dt_math
import queue
import copy
import argparse
import random
import operator


# Globals that only need to be set once per execution.
confidence_level = 0.95
is_entropy = True
most_popular_classification_global = ''

def main():
    global confidence_level
    global is_entropy

    # Parsing command line arguments such as confidence_level and the file name for the training file.
    description_string = "Anthony Galczak/Tristin Glunt\n" \
                         "CS529 - Machine Learning - Project 1 Decision Trees\n" \
                         "ID3 Decision Tree Algorithm and Random Forest implementation."
    rf_string = "Specify this option if you want to use random forests.\n" \
                "Otherwise we will build one tree only.\n" \
                "Note that building random forests takes time. Expect 300 trees to take 5-10 minutes."
    parser = argparse.ArgumentParser(description=description_string, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', dest='confidence_level', type=float, action='store', nargs='?',
                        default=0.95, help='Confidence level of chi square. Acceptable values are 0.90, 0.95, 0.99, 0')
    parser.add_argument('-i', dest='impurity_method', type=str, action='store', nargs='?',
                        default="entropy", help='Which impurity method do you want. Acceptable values are entropy, ent, gni.')
    parser.add_argument('-t', dest='training_file', type=str, action='store', nargs='?',
                        default="training.csv", help='Specify the training file you want to use. Default is \"training.csv\"')
    parser.add_argument('-r', dest='testing_file', type=str, action='store', nargs='?',
                        default="testing.csv", help='Specify the testing file you want to use. Default is \"testing.csv\"')
    parser.add_argument('-o', dest="output_file", type=str, action='store', nargs='?',
                        default="output.csv", help='Specify where you want your output of classifications to go. Default is \"output.csv\"')
    parser.add_argument("-f", "--rf", action='store_true', help=rf_string)
    args = parser.parse_args()

    confidence_level = args.confidence_level

    if confidence_level not in [0.90, 0.95, 0.99, 0]:
        print("Improper confidence level specified. Using default confidence of 0.95.")
        confidence_level = 0.95

    impurity_string = (args.impurity_method).lower()

    # Using the argument to select whether we are using gni or entropy
    if impurity_string == "ent" or impurity_string == "entropy":
        is_entropy = True
    elif impurity_string == "gni":
        is_entropy = False

    partition_size = 1

    status_print(confidence_level, args.training_file, args.testing_file, args.output_file, args.rf)

    # Running standard decision tree
    if args.rf == True:
        list_of_decision_trees = train_rf(args.training_file, partition_size)
        test_rf(list_of_decision_trees, args.testing_file, partition_size, args.output_file)
    # Running random forests, numerous decision trees.
    else:
        decision_tree = train(args.training_file, partition_size)
        test(decision_tree, args.testing_file, partition_size, args.output_file)


# status_print()
# Prints out to the user what they have selected as options for running this program.
def status_print(confidence_level, training_file, testing_file, output_file, rf):
    print("You have selected the following options:")
    print("Confidence level: %f" % confidence_level)
    if is_entropy == True:
        impurity_string = "Entropy"
    else:
        impurity_string = "GNI Index"
    print("Impurity Method: %s" % (impurity_string))
    print("Training data file name: %s" % training_file)
    print("Testing data file name: %s" % testing_file)
    print("Prediction output file name: %s" % output_file)
    if rf == True:
        print("Random forests.")
    else:
        print("Single ID3 decision tree.")

    print("\n")


# train_rf()
# Random Forests' collection method for running against training data and building decision trees.
def train_rf(training_file_name, partition_size):

    num_of_trees = 100
    list_of_data = []
    list_of_data_features_split = []
    list_of_features = []
    list_of_decision_trees = []

    data = load_file(training_file_name)
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)

    # Gathering random sets of data and features
    for x in range(num_of_trees):
        num_of_elements = random.randint(0, 600) + 400
        np.random.shuffle(data_features_split) #shuffle data so range of 400-800 is always different data

        #random features
        num_of_features = random.randint(15, 45)
        np.random.shuffle(feature_objects)

        #the shuffled data for each tree, will have as many shuffled sets of data as there are trees
        list_of_data.append(data_features_split[:num_of_elements, :])
        list_of_features.append(feature_objects[:num_of_features])

    print("Training on %d trees." % num_of_trees)

    #for each tree, get the list of classifictions and pass ID3 a random subset of data and features
    for x in range(num_of_trees):
        list_of_classes = get_classifications(list_of_data[x][:,-1:])
        decision_tree = ID3(list_of_data[x], list_of_classes, list_of_features[x])
        list_of_decision_trees.append(decision_tree)
        print(x) # This prints out which tree we are on

    return list_of_decision_trees


# test_rf()
# Random Forests' collection method for running against testing data
def test_rf(decision_trees, testing_file_name, partition_size, output_file_name):

    testing_data = load_file(testing_file_name, None)
    test_features_split = split_features(testing_data, partition_size, False)

    #list of predictions will be a list of lists for each trees predictions
    list_of_predictions = []

    #for every decision tree, make a list of predictions on testing data
    length_of_decision_trees = len(decision_trees)
    for i in range(length_of_decision_trees):
        testing_predictions = predict(decision_trees[i], test_features_split, 2001)
        list_of_predictions.append(testing_predictions)

    most_popular_predictions = []

    #inefficient loop. looping cols x rows
    for j in range(len(testing_data)):

        class_dict = {}
        class_dict['N'] = 0
        class_dict['IE'] = 0
        class_dict['EI'] = 0

        #loop through each decision trees output for this column
        for i in range(length_of_decision_trees):

            if list_of_predictions[i][j] == None or list_of_predictions[i][j][0] == 'N':
                class_dict['N'] = class_dict['N'] + 1
            elif list_of_predictions[i][j][0] == 'IE':
                class_dict['IE'] = class_dict['IE'] + 1
            elif list_of_predictions[i][j][0] == 'EI':
                class_dict['EI'] = class_dict['EI'] + 1

        # Looks at all items in a dictionary and grabs the key corresponding to the max value
        max_class = max(class_dict.items(), key=operator.itemgetter(1))[0]

        #set most_popular_predictions
        most_popular_predictions.append((max_class, list_of_predictions[i][j][1]))

    output_predictions(most_popular_predictions, output_file_name)


# train()
# Collection method for building an ID3 tree with training data.
def train(training_file_name, partition_size):
    global most_popular_classification_global
    print("Loading file: %s" % training_file_name)
    data = load_file(training_file_name)
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    list_of_classes = get_classifications(data_features_split[:,-1:])
    most_popular_classification_global = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
    decision_tree = ID3(data_features_split[:, :], list_of_classes, feature_objects)
    return decision_tree


# test()
# Collection method for classifying testing data against our decision tree that
# we built earlier.
def test(decision_tree, testing_file_name, partition_size, output_file_name):
    print("Loading file: %s" % testing_file_name)
    testing_data = load_file(testing_file_name, None)
    test_features_split = split_features(testing_data, partition_size, False)
    testing_predictions = predict(decision_tree, test_features_split[:,:], 2001)
    print("Outputting to file: %s" % output_file_name)
    output_predictions(testing_predictions, output_file_name)


# load_file()
# Loads a particular csv file and returns it as a pandas data frame.
def load_file(file_name, header_size = 1):
    try:
        file = pd.read_csv(file_name, header = header_size)
    except:
        print("ERROR: We had a problem opening your file. Try again.")
        exit(0)
    data = file.values
    return data


# split_features()
# Splits the string of features every nth character denoted by partition_size.
# Either returns a list of features or a list of features plus classifications.
def split_features(data, partition_size = 1, is_training = True):
    features = data[:, 1]
    matrix_of_features = []

    for sequence in features:
        split_sequence = [sequence[i:i+partition_size] for i in range(0, len(sequence), partition_size)]
        matrix_of_features.append(split_sequence)

    # Concatenation of features and the classifications
    if is_training == True:
        return np.c_[matrix_of_features, data[:, 2]]
    # If we aren't training, then we won't have classifications to return
    else:
        return np.c_[matrix_of_features]


# create_features()
# Goes through all of our training data and creates feature and branch objects
# corresponding to our features and their child values (A,C,G,T, etc.).
def create_features(data_features_split):
    list_of_features = []

    #go through each feature in data
    for i in range(0, data_features_split.shape[1]-1):
        feature = dt.Feature(i, [])

        # For every entry of data, add a branch to a given feature if it isn't
        # already in the list of branches
        for example in data_features_split:
            example_value = example[i]
            branch_list = feature.get_branches()
            found_example_value = False

            for branch in branch_list:
                if branch.get_branch_name() is example_value:
                    found_example_value = True
                    break

            if found_example_value == False:
                feature.add_branch(example_value)

        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features


# get_classifications()
# Obtaining the classifications from our data. For the DNA data, should be ["IE", "EI", "N"]
def get_classifications(class_list):
    classes = set()
    for list in class_list:
        classes.add(list[0])

    list_of_classes = []
    for element in classes:
        list_of_classes.append(element)

    return list_of_classes


# ID3()
# The primary algorithm of this program. ID3 is the decision tree code that navigates and
# builds up the tree. There are base cases to catch when only one classification is
# remaining or if we ran out of features (nodes) to place in the tree.
# From there, it recurses back into an ID3 call to continue traversing down the tree.
#
# From the book, our parameters match up to theirs as such:
# "data_features_split" is examples, "list_of_classes" is target_attribute, "attributes" is feature_objects
def ID3(data_features_split, list_of_classes, feature_objects):
    data_features_split_copy = copy.deepcopy(data_features_split)
    feature_objects_copy = copy.deepcopy(feature_objects)

    # there are multiple base cases in which we return a classification
    base_case = "None"
    base_case = id3_base_cases(data_features_split, list_of_classes, feature_objects)
    if base_case != "None":
        return base_case

    # "The attribute from Attributes that best* classifies Examples", use info gain to
    # determine the best feature
    highest_ig_feature_index, highest_ig_num = get_highest_ig_feat(data_features_split_copy, feature_objects_copy, list_of_classes)

    # check if we're using chi-square or not
    if confidence_level != 0:
        current_feature = feature_objects_copy[highest_ig_feature_index]
        #determine if this feature will be of statistical benefit using chi-square
        feature_is_beneficial = dt_math.chi_square_test(data_features_split, current_feature, list_of_classes, confidence_level)

        #if the feature is not beneficial, return a leaf node of the most popular class
        if not feature_is_beneficial:
            most_common_class = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
            return most_common_class

    node = feature_objects_copy[highest_ig_feature_index]
    feature_objects_copy[highest_ig_feature_index] = "None"

    # For every possible branch(value). Should look like {A, C, G, T}
    for branch in node.get_branches():
        # Gathering all examples that match this branch value, returns a numpy matrix
        subset_data_feature_match = np.array(dt_math.get_example_matching_value(data_features_split_copy, branch.get_branch_name(), node))

        # If the examples list is empty(ie., there are no examples left that have this value after trimming so many subsets)
        if subset_data_feature_match.shape[0] == 0:
            most_common_class = dt_math.determine_class_totals(data_features_split_copy, list_of_classes, True)
            branch.add_child_feature(most_common_class)
        else:
            # Recurse
            #this is being applied to the very first instance of feature objects in the first call when feature 29 is parent
            child_feature = ID3(subset_data_feature_match, list_of_classes, feature_objects_copy)
            branch.add_child_feature(child_feature)

    return node

# id3_base_cases() checks each base case for the ID3 algorithm. If any are true,
# this function will return a leaf that's a classification. If none are, it returns "None"
def id3_base_cases(data_features_split_copy, list_of_classes, feature_objects_copy):

    # If all of the remaining examples have the same classification, return that classification
    initial_classification = "None"
    found_different = False
    for example in data_features_split_copy:

        #classification equals the last element in our example
        classification = example[-1:]

        if initial_classification == "None":
            initial_classification = classification
        elif classification != initial_classification:
            found_different = True
            break

    if found_different is False:
        return classification

    #if there are no more features to look at, return with a leaf of the most common class
    #this is really ugly, but since we're not removing things for the list, not much else to do
    all_features_used = True
    for feature in feature_objects_copy:
        if feature != "None":
            all_features_used = False
            break

    if all_features_used:
        print("All features have been used, returning most common class")
        most_common_class = dt_math.determine_class_totals(data_features_split_copy, list_of_classes, True)
        return most_common_class

    return "None"

# get_highest_ig_feat()
# Obtaining the highest information gain feature index from the remaining list of features
def get_highest_ig_feat(data_features_split, feature_objects, list_of_classes):

    list_of_igs = []

    # Getting how many characters long each example is
    length_of_data = data_features_split.shape[1] - 1

    highest_ig_num = 0.0
    highest_ig_index = -1

    for feature_index in range(len(feature_objects)):
        if feature_objects[feature_index] != "None":
            info_gained_entropy = dt_math.gain(data_features_split, feature_objects[feature_index], list_of_classes, is_entropy)

            # Getting the highest info gained feature
            if info_gained_entropy > highest_ig_num:
                highest_ig_num = info_gained_entropy
                highest_ig_index = feature_index

    #if highest info gain is 0, return arbitrary index
    if highest_ig_num == 0:
        for feature_index in range(length_of_data):
            if feature_objects[feature_index] != "None":
                print("Highest_ig_num was 0, returning random feature index: " + str(feature_index))
                highest_ig_index = feature_index
                break

    # Outputting the index of the feature that has the highest info gained
    return highest_ig_index, highest_ig_num


# predict()
# Traverses the decision tree that we made from training in order to classify
# new data. Returns a full list of predictions (classifications) for a set of data.
def predict(decision_tree, data, data_index):
    global most_popular_classification_global
    predictions = []
    node = decision_tree

    """rf edge case addition """
    if type(node) == str:
        for example in data:
            data_index += 1
            predictions.append((node, data_index))
        return predictions

    for example in data:
        #get this current features value at the index of the current node of the tree
        current_feature_data_value = example[node.feature_index]
        #go through the possible values for this feature
        for branch in node.get_branches():
            #if the current value of the feature in this example is equal to a branches value
            if current_feature_data_value == branch.branch_value:
                #if the child feature of this branch is a leaf
                if type(branch.child_feature) is not dt.Feature:
                    if type(branch.child_feature) is not str:
                        temp_prediction = branch.child_feature[0]
                    else:
                        temp_prediction = branch.child_feature

                    tuple_prediction = (temp_prediction, data_index)
                    predictions.append(tuple_prediction)
                    node = decision_tree
                    data_index += 1
                    break
                #if the child feature is not a leaf
                else:
                    node = branch.child_feature
                    recursive_prediction = recursive_prediction_traversal(example, node, data_index)

                    # If we got to a recursion step and it returned no classification, then
                    # assign it to the most popular classification so far.
                    if recursive_prediction == None:
                        recursive_prediction = (most_popular_classification_global, data_index)

                    predictions.append(recursive_prediction)
                    node = decision_tree
                    data_index += 1
                    break

    return predictions

# recursive_prediction_traversal()
# takes in the current example being predicted on and recursively iterates down the tree
# until a branch is met that has a classification is its child. This should all be one
# predict method but, I had a mindblock writing the entire predict function recursively
# when we had to go through each example. Thus, the predict function handles going through
# each example while this recursive function goes all the way down the tree on one example
def recursive_prediction_traversal(single_example, node, data_index):
    current_feature_data_value = single_example[node.feature_index]

    # go through each value for this current feature
    for branch in node.get_branches():
        # if the current value for this feature from the data entry is equal to the trees branch value
        if current_feature_data_value == branch.branch_value:
            #if the child of this branch isn't another feature, we've found a classificaiton
            if type(branch.child_feature) is not dt.Feature:
                if type(branch.child_feature) is not str:
                    temp_prediction = branch.child_feature[0]
                else:
                    temp_prediction = branch.child_feature

                tuple_prediction = (temp_prediction, data_index)
                return tuple_prediction
            #if the child feature is a feature we need to recurse down a level
            else:
                node = branch.child_feature
                return recursive_prediction_traversal(single_example, node, data_index)

# output_predictions()
# Takes our predictions and outputs them to a file for eventual submission.
def output_predictions(predictions, file_name):
    try:
        file = open(file_name, "w")
    except:
        print("ERROR: We had a problem writing to this file. Try again.\n")
        exit(0)
    file.write("ID,Class\n")
    for tuple in predictions:
        #print(tuple)
        file.write(str(tuple[1]) + "," + str(tuple[0]) + "\n")

    file.close()


if __name__ == "__main__":
    main()


# Helper/Debug Methods

# traverse_tree()
# BFS-like algorithm to go through decision tree. This is used purely for
# debugging purposes to see how the tree is being built/what it looks like.
def traverse_tree(decision_tree):
    q = queue.Queue()

    #insert root feature into queue
    q.put(decision_tree)

    while not q.empty():
        v = q.get()

        if type(v) is dt.Feature:
            print("\nFeature: " + str(v.feature_index))

            for branch in v.get_branches():
                #print(branch.get_branch_name() + " ", end='')
                q.put(branch)
            #print("")
        elif type(v) is dt.Branch:
            print("\nValue:" + str(v.get_branch_name()))

            if type(v.child_feature) is str or type(v.child_feature) is np.ndarray:
                print("---> Leaf: " + str(v.child_feature))
            elif type(v.child_feature) is dt.Feature:
                print("---> Child Feature: " + str(v.child_feature.feature_index))

            q.put(v.child_feature)
        #elif type(v) is str:
            #print("Leaf: " + str(v))


# run_validation_dataset()
# This method is useful for testing against a known validation dataset instead of
# depending upon verifying our testing dataset against Kaggle.
def run_validation_dataset(data_features_split, list_of_classes, feature_objects, start_index):
    decision_tree = ID3(data_features_split[:1598, :], list_of_classes, feature_objects)

    predictions = predict(decision_tree, data_features_split[1598:, :], 1598)

    correct = 0
    wrong = 0
    for num in range(400):#data_features_split[1598:, :]:
        actual_class = data_features_split[1598 + num, -1]
        predicted_class = predictions[num][0]

        if actual_class == predicted_class:
             correct += 1
        else:
            wrong += 1

    print("Accuracy: %f" % (correct / (correct + wrong)))
