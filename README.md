# ML_Project_2017-2018
Decision Trees - CART for classification problems

The aim of this project is to expose the main features of one of the commonly used Decision Tree algorithm: the CART algorithm. The project is developed in python and offers a basic CART implementation and applications on some synthetic data sets and UCI data sets for classification problems.

## Project Structure
src folder
* cart.py contains the implementation of main procedures for CART
* tree.py contains the tree class declarations
* utilities.py and plot_utilities contain some basic useful functions 
* simulations.py contains a main function to run four simulations on some synthetic data sets
* banknote.py and parkinson.py contain two main functions to run CART on two UCI data sets

dataset folder
* contains the synthetic and UCI data sets prepared for CART

## Reproduce my experiments
To reproduce my experiments just download the project and run with python (or Pycharm IDE) one file among simulations.py, banknote.py or parkinson.py as main file:

* simulations.py : the main function runs automatically my four simulations on synthetic data sets and shows the related charts.

* bankonte.py: the main function runs automatically an hold out cross validation on the UCI banknote authentication data set.
The execution with the current configuration is very time-consuming ( a couple of hours on my laptop) but you can decrease the number of repetitions modifying some highlighted variables in the main function.
You can also directly visualize the chart of my results (without executing the algorithm) defining the main function as follow:

def main():
    plot_only_results()
 
* parkinson: the main function runs automatically a nested cross validation on the UCI parkinson data set.
The execution with the current configuration takes more or less one hour but you can modify some variables as before in order to decrease the execution time. However, there is not a plot_only_results() function defined for this file.

## Author

* **Stefano Rebora** - *Università degli studi di Genova - ML 2017-2018* 
