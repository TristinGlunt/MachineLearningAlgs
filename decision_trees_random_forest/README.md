This is the README for UNM CS529 Machine Learning Project 1.
It includes a command line interface for execution.

This was written by:
Anthony Galczak, Tristin Glunt
agalczak@unm.edu, tglunt@unm.edu

Required libraries for execution:
Python 3.6.0 & Anaconda 4.3.0 (pandas)

Brief summary on the files contained in this program:
decision_tree.py
This holds our classes needed to build a decision tree. We use "Feature" to
map to a node and "Branch" to map to a branch. The architecture of a decision
tree requires the branch to have a value, hence a Branch class.

dt_math.py
This holds our math libraries that we wrote for execution of the decision tree.
You will find things like impurity(entropy/gni), chi_square and gain in here.

main.py
This holds the bulk of the code including the ID3 algorithm code. There is
multiple paths of execution based upon the command line arguments as we have
additionally implemented random forests.

How to run and use our program:
Our main file for execution is "main.py".
There's two paths to the naive execution of the file.
"python main.py" - Make sure your $PATH includes Python3.6 and Anaconda 4.3.0.
"./main.py" - Make sure main.py has execute bit set for your user.

It really doesn't matter which method you use, but just make sure that you have
the correct libraries and Python version to be able to run the file.

By running this naive execution, you will run the default settings of the
program, which are the following:
Confidence_level: 0.95, Impurity_method: Entropy, Training_file: training.csv,
Testing_file: testing.csv, Output_file: output.csv, no random forests (1 tree).

All of these parameters are configurable via the command line interface.
If you type "python main.py --help" you will receive a help menu like this:


usage: main.py [-h] [-c [CONFIDENCE_LEVEL]] [-i [IMPURITY_METHOD]]
               [-t [TRAINING_FILE]] [-r [TESTING_FILE]] [-o [OUTPUT_FILE]]
               [-f]

Anthony Galczak/Tristin Glunt
CS529 - Machine Learning - Project 1 Decision Trees
ID3 Decision Tree Algorithm and Random Forest implementation.

optional arguments:
  -h, --help            show this help message and exit
  -c [CONFIDENCE_LEVEL]
                        Confidence level of chi square. Acceptable values are 0.90, 0.95, 0.99, 0
  -i [IMPURITY_METHOD]  Which impurity method do you want. Acceptable values are entropy, ent, gni.
  -t [TRAINING_FILE]    Specify the training file you want to use. Default is "training.csv"
  -r [TESTING_FILE]     Specify the testing file you want to use. Default is "testing.csv"
  -o [OUTPUT_FILE]      Specify where you want your output of classifications to go. Default is "output.csv"
  -f, --rf              Specify this option if you want to use random forests.
                        Otherwise we will build one tree only.
                        Note that building random forests takes time. Expect 300 trees to take 5-10 minutes.


This help menu should guide you through any questions about the parameter usage,
but I will display a few examples for edification.


python main.py -c 0.95 -i entropy -t training.csv -r testing.csv -o output.csv
The above will run the default settings that you'll get with no flags set.

python main.py -c 0.99 -i gni -t training.csv -r testing.csv -o rf_test.csv -f
This will run the program with 0.99 confidence, gni impurity function, loading
training file from "training.csv", loading testing file from "testing.csv",
outputting your data to "rf_test.csv" and will run random forests.

python main.py -c 0.90 -i gni -o test.csv -f
The above will run 0.90 confidence, gni impurity, output to "test.csv", and run
random forests. The rest will be set to defaults.


Please note that this CLI isn't hardened against a very determined user. If you
try hard to make it error, it will fail. It has been hardened against very basic
things like entering invalid confidence values (will set to default 0.95) or
entering a bogus purity_method string. Things like file permissions and hooks
are well beyond the scope of this program and project and are not protected against.


Please note; we have not setup the cli options to configure the random forests
programmatically. This isn't difficult at all to implement, but we decided
this would be a bit overboard for cli options on a program like this. If you
would like to play around with the random forests hyperparameters they are
available in main.py/train_rf(). The parameters you would want to play with are:
num_of_trees, num_of_elements (size of data), num_of_features.
