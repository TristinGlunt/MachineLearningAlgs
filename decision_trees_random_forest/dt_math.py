# Tristin Glunt / Anthony Galczak - tglunt@unm.edu - agalczak@unm.edu
# CS529 - Project 1 - Decision Trees

import math
import numpy as np

# The first index is degrees of freedom, the key afterwards is the confidence level
# degrees of freedom should be max 10, max values = 6, max classes = 3 (3-1)*(6-1) = 10
chi_square = [0, {0.90 : 2.71, 0.95 : 3.84, 0.99 : 6.63}, {0.90 : 4.61, 0.95 : 5.99, 0.99 : 9.21},
{0.90 : 6.25, 0.95 : 7.81, 0.99 : 11.34}, {0.90 : 7.78, 0.95 : 9.49, 0.99 : 13.28}, {0.90 : 9.24, 0.95 : 11.07, 0.99 : 15.09},
{0.90 : 10.64, 0.95 : 12.59, 0.99 : 16.81}, {0.90 : 12.02, 0.95 : 14.07, 0.99 : 18.48}, {0.90 : 13.36, 0.95 : 15.51, 0.99 : 20.09},
{0.90 : 14.68, 0.95 : 16.92, 0.99 : 21.67}, {0.90 : 15.99, 0.95 : 18.31, 0.99 : 23.21}]


# impurity()
# This is the combined functionality of entropy and gni_index.
# It gets the proportionality of each classification and then loops over
# all of the classes to calcuulate the impurity value. If is_entropy is True,
# then the math will be entropy, otherwise the math will be gni_index.
def impurity(examples, classes, is_entropy):

    # Starting value for entropy is 0, 1 for gni_index
    if is_entropy == True:
        impurity_value = 0
    else:
        impurity_value = 1

    # Getting the proportionality for each classification
    label_totals = determine_class_totals(examples, classes)

    total_examples = len(examples)
    total_classes = len(classes)

    if total_examples == 0:
        return 0

    # Loop over all possible classifications and calculate the purity via either
    # entropy or gni_index, depending on which was selected.
    for i in range(total_classes):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            # Entropy
            if is_entropy == True:
                impurity_value = impurity_value - (p_i * math.log(p_i, 2))
            # GNI Index
            else:
                impurity_value = impurity_value - (p_i ** 2)

    return impurity_value


# gain()
# Gain calculates the information gain of each feature on current passed in examples
# gain(data, A) -> will look through values Y & Z for feature A.
# feature - an object... with this features column index in the
# dataset and needs to have a list of it's values.
def gain(examples, feature, classes, is_entropy):
    #determine impurity of entire dataset
    gain = impurity(examples, classes, is_entropy)

    #determine impurity for each value in feature, sum together and return info gain
    for branch in feature.get_branches():

        #return subset of examples that only have this value
        subset_of_example = get_example_matching_value(examples, branch.get_branch_name(), feature)

        total_subset_of_value = len(subset_of_example)

        #math of information gain
        proportion_of_subset = total_subset_of_value / len(examples) * 1.0
        subset_entropy = proportion_of_subset * impurity(subset_of_example, classes, is_entropy)
        gain = gain - subset_entropy

    return gain


# determine_class_totals takes in a 2D data set, a list of possible classes, and a boolean
# flag. If the boolean flag is set to false, will return the total occurrences for each class
# in the data. If true, will only return whichever Class occurred the most.
def determine_class_totals(examples, classes, get_most_common_class = False):
    labels = {}
    label_totals = {}

    numOfClasses = len(classes)

    #initialize each class in the dictionary
    for i in range(numOfClasses):
        labels["class" + str(i)] = classes[i]

    #initialize the total count for each class
    for i in range(numOfClasses):
        label_totals["class" + str(i)] = 0

    #go through each example
    for example in examples:
        #go through each class for current example, once match found, break
        for i in range(numOfClasses):
            #the output will always be the last element of the example
            if(example[-1:] == labels["class" + str(i)]):
                #if the output class for this example matches, add one to total classes
                label_totals["class" + str(i)] = label_totals["class" + str(i)] + 1
                break

    most_common_class_amount = 0
    most_common_class = ""

    # if wanting only the most popular class
    if get_most_common_class:
        for i in range(numOfClasses):
            if label_totals["class" + str(i)] > most_common_class_amount:
                most_common_class_amount = label_totals["class" + str(i)]
                most_common_class = labels["class" + str(i)]
        return most_common_class
    else:
        # return all counts for each class
        return label_totals

# get_example_matching_value() takes in a dataset, the current name for the value we're
# delimiting on, and the feature where this value occurs, so we only look at that feature
# in the data. This will return a subset of the data where it only has the given value
# (branch_name) for the passed in feature. Mainly used by info_gain when looping through
# values for a feature and calculating each values impurity.
def get_example_matching_value(examples, branch_name, feature):
    subset_of_value = []

    #go through each example
    for example in examples:
        #go through only feature passed into gain and check value
        # if the value for this feature matches, take this entire example into the subset
        if(example[feature.feature_index] is branch_name):
            subset_of_value.append(example)

    #convert subset to np matrix so we can use .shape
    subset_of_value = np.array(subset_of_value)
    return subset_of_value

# chi_square_test() is the implementation of the chi_square function and the table. This
# function helps ID3 not overfit the data by introducing a statistical early-stopping
# method for when the next feature is not beneficial up to a certain confidence_level
def chi_square_test(data, current_feature, list_of_classes, confidence_level):
    #determine the class totals for this entire feature
    class_totals = determine_class_totals(data, list_of_classes, False)
    #calculate the needed table for chi-square
    variable_matrix_real, variable_matrix_expected = build_variable_matrix_tables(data, current_feature, list_of_classes, class_totals)

    chi_square_value = 0

    #for every value in this feature
    for i in range(len(current_feature.get_branches())):
        #for every class this value has
        for j in range(len(class_totals)):
            #don't divide by 0
            if variable_matrix_expected[i][j] == 0:
                continue
            chi_square_value += ((variable_matrix_real[i][j] - variable_matrix_expected[i][j]) ** 2) / variable_matrix_expected[i][j]

    # determine the degree_of_freedom for this feature
    degree_of_freedom = (len(list_of_classes) - 1)  * (len(current_feature.get_branches()) - 1)

    #compute critical_value by using the chi_square distribution table
    critical_value = chi_square[degree_of_freedom][confidence_level]

    # Evaluating whether or not the chi_square_value is within the confidence level
    if chi_square_value < critical_value:
        # this feature is not statistically beneficial
        return False
    else:
        return True

# build_variable_matrix_tables() computes the table needed for chi_square. This table
# counts the number of real appearances for a particular value and the possible classes
# for that value. It also calculates the expected number of appearances of a value with
# a classification. These two matrices are used by the chi-square function
def build_variable_matrix_tables(data, current_feature, list_of_classes, class_totals):

    total_data_size = len(data)

    #build a zero matrix the dimensions of, (total_values_for_feature, total_classes)
    variable_matrix_real = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])
    variable_matrix_expected = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])

    #determine "real values" for each value and class of this feature
    counter = 0
    for branch in current_feature.get_branches():
        #returns subset of data matching the current value of this feature
        subset_data_feature_match = get_example_matching_value(data, branch.get_branch_name(), current_feature)

        #returns a dictionary of totals of each class for this value of this feature
        class_totals_for_subvalue = determine_class_totals(subset_data_feature_match, list_of_classes, False)

        #fill variable_matrix_real with class totals for this current branch
        for j in range(0, len(class_totals_for_subvalue)):
            variable_matrix_real[counter][j] = class_totals_for_subvalue["class" + str(j)]
        counter += 1

    #calculate expected values
    counter = 0
    for branch in current_feature.get_branches():
        # returns subset of data matching the current value of this feature
        subset_data_feature_match = get_example_matching_value(data, branch.get_branch_name(), current_feature)

        # get the number of total appearances of this value for this feature
        total_subset_size = len(subset_data_feature_match)

        # get the total of each class for this subset
        class_totals_for_subvalue = determine_class_totals(subset_data_feature_match, list_of_classes, False)

        #calculate expected values for every class for this current value
        for j in range(len(class_totals_for_subvalue)):
            variable_matrix_expected[counter][j] = total_subset_size * (class_totals["class" + str(j)] / total_data_size)
        counter += 1

    return variable_matrix_real, variable_matrix_expected
