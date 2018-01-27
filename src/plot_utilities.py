
# GENERAL PLOT UTILITIES

import matplotlib.pyplot as plt

from src.tree import Node


# Plot a 2D classification dataset
def plot_2D_dataset(dataset, title):
    plt.figure(title)
    X = [row[0] for row in dataset]
    Y = [row[1] for row in dataset]
    labels = [row[2] for row in dataset]
    plt.scatter(X, Y, marker='o', c=labels)


# Print separating lines for 2D classification dataset
def print_tree_separating_2D(node):
    if isinstance(node, Node):
        if node.index == 0:
            plt.axvline(node.val)
        else:
            plt.axhline(node.val)
        print_tree_separating_2D(node.left)
        print_tree_separating_2D(node.right)


def show_plot():
    plt.draw()
