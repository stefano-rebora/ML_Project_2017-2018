
# CART ALGORITHM IMPLEMENTATION

from src.tree import Tree, Node


# Split a dataset based on an attribute and an attribute value
def make_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Compute the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset and return the related node
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    split_index, split_value, split_score, split_groups = 0, 0, 999, None
    # Iterate over each attribute (over columns)
    for index in range(len(dataset[0]) - 1):
        # Sort data (rows) by the selected attribute
        dataset.sort(key=lambda x: x[index])
        # Iterate over each possible attribute value (over rows)
        for i in range(0, len(dataset) - 1):
            value = (dataset[i][index] + dataset[i + 1][index]) / 2
            groups = make_split(index, value, dataset)
            gini = gini_index(groups, class_values)
            if gini < split_score:
                split_index, split_value, split_score, split_groups = index, value, gini, groups
    return Node(split_index, split_value, split_groups)


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    # Select the most common class value in the group
    return max(set(outcomes), key=outcomes.count)


# Check if a node is pure (all values of the same class)
def is_a_pure_node(group):
    for i in range(0, len(group)):
        if group[0][-1] != group[i][-1]:
            return False
    return True


# Grow the tree creating child splits for a node or making terminal nodes
def grow_tree(node, max_depth, min_size, depth):
    left, right = node.groups
    del node.groups
    # check for a no split
    if not left or not right:
        node.left = node.right = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node.left, node.right = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size or is_a_pure_node(left):
        node.left = to_terminal(left)
    else:
        node.left = get_split(left)
        grow_tree(node.left, max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size or is_a_pure_node(right):
        node.right = to_terminal(right)
    else:
        node.right = get_split(right)
        grow_tree(node.right, max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    tree = Tree(get_split(train))
    grow_tree(tree.root, max_depth, min_size, 1)
    return tree


# Make a prediction with a decision tree
def predict(node, row):
    if row[node.index] < node.val:
        if isinstance(node.left, Node):
            return predict(node.left, row)
        # If the left child is a terminal node
        else:
            return node.left
    else:
        if isinstance(node.right, Node):
            return predict(node.right, row)
        # if the right child is a terminal node
        else:
            return node.right


# Get the total number of nodes in the tree
def get_nodes_number(root):
    if not isinstance(root, Node):
        return 1
    else:
        return 1 + get_nodes_number(root.left) + get_nodes_number(root.right)


# CART Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree.root, row)
        predictions.append(prediction)
    return predictions


# Print a decision tree
def print_tree(node, feature_names, depth=0):
    if isinstance(node, Node):
        print('%s[%s%d < %.3f]' % (depth * ' ', feature_names[node.index], (node.index + 1), node.val))
        print_tree(node.left, feature_names, depth + 1)
        print_tree(node.right, feature_names, depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))
