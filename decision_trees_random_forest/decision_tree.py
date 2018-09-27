# Tristin Glunt / Anthony Galczak - tglunt@unm.edu - agalczak@unm.edu
# CS529 - Project 1 - Decision Trees

# A given feature is a node, the values are otherwise known as "branches"
class Feature():

    def __init__(self, feature_index, branches):
        self.feature_index = feature_index
        self.branches = branches

    def add_branch(self, branch_value):
        self.branches.append(Branch(branch_value))

    def get_branches(self):
        return self.branches

# This is a branch in a tree that also corresponds to values in the
# given decision tree algorithms (ID3).
class Branch():
    def __init__(self, branch_value):
        self.branch_value = branch_value

    # What feature is this branch pointing to
    def add_child_feature(self, feature):
        self.child_feature = feature

    def get_branch_name(self):
        return self.branch_value
