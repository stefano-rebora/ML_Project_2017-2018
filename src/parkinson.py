
# CART APPLIED ON PARKINSON VOICE RECORDING DATASET
# Run this file as main to execute cart on parkinson dataset

from statistics import mean, stdev

from utilities import load_csv, dataset_to_float, two_grid_search_with_accuracy_return, accuracy_metric, \
    cross_validation_split

from src.cart import decision_tree


# Run CART on parkinson dataset with a nested k-fold cross validation
def parkinson_main(n_folds_outer_cross_val, n_folds_inner_cross_val, max_depth, min_size):

    if n_folds_outer_cross_val < 2 or n_folds_inner_cross_val < 2:
        raise ValueError("Illegal value parameter")

    filename = '../dataset/parkinson_recording_data.csv'
    dataset = load_csv(filename)
    dataset = dataset_to_float(dataset)
    folds = cross_validation_split(dataset,n_folds_outer_cross_val)
    scores = list()
    outer_fold_number = 0

    # Outer k-fold cross validation
    for fold in folds:
        outer_fold_number += 1
        # Prepare train and test set
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        # Inner k-fold cross validation ( grid search )
        best_couple, inner_accuracy = two_grid_search_with_accuracy_return(decision_tree, n_folds_inner_cross_val, train_set, max_depth, min_size)
        # Evaluate results on outer cross validation test set
        predictions = decision_tree(train_set, test_set, best_couple[0], best_couple[1])
        actual = [row[-1] for row in fold]
        outer_accuracy = accuracy_metric(actual, predictions)
        print("-" * 10 + " Outer Fold n. " + str(outer_fold_number) + " " + "-" * 10)
        print("Best params selected by inner cross validation (max_depth,min_size): "+str(best_couple[0])+" "+str(best_couple[1]))
        print("Best params mean accuracy in the inner cross validation: " + str(inner_accuracy))
        print("Best params accuracy in the outer cross validation: " + str(outer_accuracy))
        scores.append(outer_accuracy)

    print("-" * 10 + " Final Results " + " " + "-" * 10)
    print("Total Accuracy mean: " + str(mean(scores)))
    print("Total Accuracy std dev: " + str(stdev(scores)))
    return scores


def main():
    n_folds_outer_cross_val = 5     # number of folds for outer cross validation
    n_folds_inner_cross_val = 5     # number of folds for inner cross validation ( for two grid search)
    max_depth = list(range(1, 11))  # parameters list for the tree max depth
    min_size = list(range(1, 11))   # parameters list for the tree node min size
    parkinson_main(n_folds_outer_cross_val, n_folds_inner_cross_val, max_depth, min_size)


if __name__ == "__main__":
    main()
