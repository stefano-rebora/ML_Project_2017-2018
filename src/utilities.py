
# GENERAL UTILITIES

from random import seed
from random import randrange
from csv import reader
import itertools

# To reproduce random experiments
seed(1)


# Load a CSV file
def load_csv(filename):
    file = open(filename)
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert a string column dataset to float
def dataset_to_float(dataset):
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Split randomly a dataset into training and test set
def random_training_test_split(dataset, training_percentage):
    training_split = list()
    test_split = list()
    dataset_copy = list(dataset)
    training_size = int((len(dataset) * training_percentage) / 100)
    while len(training_split) < training_size:
        index = randrange(len(dataset_copy))
        training_split.append(dataset_copy.pop(index))
    for row in dataset_copy:
        test_split.append(row)
    return training_split, test_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Two grid search algorithm for parameter tuning, print() only for debug
def two_grid_search(algorithm, n_folds, dataset, params1, params2):
    results = list()
    couples = list(itertools.product(params1, params2))  # Get all possible couples
    for couple in couples:
        # print("Evaluating couple:"+str(couple))
        scores = evaluate_algorithm(dataset,algorithm,n_folds,couple[0],couple[1])
        results.append(sum(scores)/float(len(scores)))
        # print("Result: " + str( sum(scores)/float(len(scores))))
    best_index = results.index(max(results))
    return couples[best_index]


#  The same before but with the mean accuracy return
def two_grid_search_with_accuracy_return(algorithm, n_folds, dataset, params1, params2):
    results = list()
    couples = list(itertools.product(params1, params2))  # Get all possible couples
    for couple in couples:
        # print("Evaluating couple:"+str(couple))
        scores = evaluate_algorithm(dataset,algorithm,n_folds,couple[0],couple[1])
        results.append(sum(scores)/float(len(scores)))
        # print("Result: " + str( sum(scores)/float(len(scores))))
    best_index = results.index(max(results))
    return couples[best_index], results[best_index]