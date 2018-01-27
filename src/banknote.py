
# CART APPLIED ON BANKNOTE AUTHENTICATION DATASET
# Run this file as main to execute cart on banknote dataset

from statistics import mean, stdev

import matplotlib.pyplot as plt
from utilities import load_csv, dataset_to_float, two_grid_search, accuracy_metric
from utilities import random_training_test_split

from src.cart import decision_tree


# Plot test accuracy for increasing training split percentage
def plot_results(tr_percentages, mean_accuracy, std_dev):
    plt.plot(tr_percentages, mean_accuracy, 'r')
    plt.errorbar(tr_percentages, mean_accuracy, std_dev, color='r', linestyle='None', marker='^')
    plt.yticks(range(80,101,2))
    plt.xticks(range(0, 100, 5))
    plt.xlabel('Training split percentage')
    plt.ylabel('Test accuracy')
    plt.draw()


# Run CART on banknote dataset and plot the results
def banknote_main(tr_percentages, number_repetitions, n_folds_2grid_search, max_depth, min_size):

    if number_repetitions < 2 or n_folds_2grid_search < 2:
        raise ValueError("Illegal value parameter")

    filename = '../dataset/data_banknote_authentication.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    mean_accuracies = list()
    std_devs = list()
    # For each percentage of training split
    for percentage in tr_percentages:
        accuracies = list()
        # Repeat number_repetions times the random split and test validation for each split
        for run in range(1, number_repetitions+1):
            train, test = random_training_test_split(dataset,percentage)
            result = two_grid_search(decision_tree,n_folds_2grid_search, train, max_depth, min_size)
            predictions = decision_tree(train,test,result[0],result[1])
            actual = [row[-1] for row in test]
            accuracy = accuracy_metric(actual, predictions)
            accuracies.append(accuracy)
        print("-"*10 + " training split %" + str(percentage)+" "+"-"*10)
        print("Accuracies of training split %"+str(percentage)+" : " + str(accuracies))
        mean_acc = mean(accuracies)
        std_dev = stdev(accuracies)
        print("Accuracy mean: " + str(mean_acc))
        print("Accuracy std dev: " + str(std_dev))
        mean_accuracies.append(mean_acc)
        std_devs.append(std_dev)

    plt.figure("BankNote dataset")
    plot_results(tr_percentages, mean_accuracies, std_devs)
    plt.show()


# Plot only results without computation
def plot_only_results():
    plt.figure("Only results with #repet= 5, #2grid_search_folds= 5")
    tr_percentages = [5, 15, 30, 60, 90]
    mean_accuracies = [93.3129, 95.8355, 97.3569, 97.5592, 98.6957]
    std_devs = [2.4934, 1.5009, 1.1787, 0.9690, 0.9448]
    plot_results(tr_percentages,mean_accuracies, std_devs)


def main():
    # plot_only_results()
    tr_percentages = [5, 15, 30, 60, 90]  # splits of percentage to test
    number_repetitions = 5                # number of repetitions for each percentage split
    n_folds_2grid_search = 5              # number of folds for the two grid search parameter tuning
    max_depth = list(range(5, 11))        # parameters list for the tree max depth
    min_size = list(range(1, 11))         # parameters list for the tree node min size

    banknote_main(tr_percentages, number_repetitions, n_folds_2grid_search, max_depth, min_size)


if __name__ == "__main__":
    main()