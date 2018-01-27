
# SIMULATIONS ON SYNTHETIC DATASETS
# Run this file as main to execute the simulations

import matplotlib.pyplot as plt
from plot_utilities import plot_2D_dataset, print_tree_separating_2D, show_plot
from utilities import load_csv, dataset_to_float, accuracy_metric

from src.cart import build_tree, print_tree, get_nodes_number, predict


# Make predictions with CART algorithm and return the tree size and the lists of predicted and expected values
def decision_tree_prediction_and_size(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    size = get_nodes_number(tree.root)
    test_set = list()
    expected = list()
    for row in test:
        row_copy = list(row)
        test_set.append(row_copy)
        expected.append(row_copy[-1])
        row_copy[-1] = None
    predicted = list()
    for row in test:
        prediction = predict(tree.root, row)
        predicted.append(prediction)
    return size, predicted, expected


# Simulation n.1 : print the separating lines and the tree for a simple binary classification dataset
def sim1():
    filename = '../dataset/dataset1.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    plot_2D_dataset(dataset, "Simulation n.1")
    print("-" * 10 + " Sim.1 TREE " + "-" * 10)
    tree = build_tree(dataset, 5, 1)
    print_tree(tree.root, ['x', 'y'])
    print_tree_separating_2D(tree.root)
    show_plot()


# Simulation n.2 : print the separating lines and the tree for a simple four classes classification dataset
def sim2():
    filename = '../dataset/dataset2.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    plot_2D_dataset(dataset, "Simulation n.2")
    tree = build_tree(dataset, 5, 1)
    print("-" * 10 + " Sim.2 TREE " + "-" * 10)
    print_tree(tree.root, ['x', 'y'])
    print_tree_separating_2D(tree.root)
    show_plot()


# Simulation n.3 : print the separating lines and the tree for a more complex four classes classification dataset
def sim3():
    filename = '../dataset/dataset3.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    plot_2D_dataset(dataset, "Simulation n.3")
    tree = build_tree(dataset, 5, 1)
    print("-" * 10 + " Sim.3 TREE " + "-" * 10)
    print_tree(tree.root, ['x', 'y'])
    print_tree_separating_2D(tree.root)
    show_plot()


# Simulation n.4 : plot the chart of train accuracy and tree complexity trend ( for the sim.n3 dataset)
def sim4():
    filename = '../dataset/dataset3.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    train_accuracy = list()
    nodes_numbers = list()
    for i in range(1, 21):
        tree_size, predicted, expected = decision_tree_prediction_and_size(dataset, dataset, i, 1)
        acc = accuracy_metric(expected, predicted)
        train_accuracy.append(acc)
        nodes_numbers.append(tree_size)
    x = range(1, 21)
    plt.figure("Simulation n.4")
    line1, = plt.plot(x, train_accuracy, 'r', label='Train accuracy')
    line2, = plt.plot(x, nodes_numbers, 'g', label='Tree complexity(# nodes)')
    plt.legend(handles=[line1, line2], loc=4)
    plt.xticks(x)
    plt.xlabel('Maximum Tree Depth')
    plt.draw()


def main():
    sim1()
    sim2()
    sim3()
    sim4()
    plt.show()


if __name__ == "__main__":
    main()