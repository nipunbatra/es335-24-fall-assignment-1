import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

'''def checkTimeForDiscreteInputDiscreteOutput():
    timeForBuilding = []
    timeForPredicting = []
    valuesOfM = []
    valuesOfN = []
    # fixing N
    N = 100
    for M in range(1, 15):
        valuesOfM.append(M)

        X = pd.DataFrame({i: pd.Series(np.random.randint(
            2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size=N), dtype="category")
        begin = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)

        end = time.time()
        timeForBuilding.append((end-begin))

        begin = time.time()

        y_hat = tree.predict(X)

        end = time.time()
        timeForPredicting.append((end-begin))

    plt.plot(valuesOfM, timeForBuilding)
    plt.title(
        "Plot of number of features vs time taken to build the tree at a constant N (time for building)")
    plt.xlabel("Number of features")
    plt.ylabel("Time taken")
    plt.show()

    #fixing M

    M = 5
    for N in range(1, 30):
        valuesOfN.append(N)

        X = pd.DataFrame({i: pd.Series(np.random.randint(
            2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size=N), dtype="category")

        begin = time.time()

        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)

        end = time.time()
        timeForBuilding.append((end-begin))

        begin = time.time()

        y_hat = tree.predict(X)

        end = time.time()
        timeForPredicting.append((end-begin))

    plt.plot(valuesOfN, timeForBuilding)
    plt.title(
        "Plot of number of data points vs time taken to build the tree at a constant M (timeForBuilding)")
    plt.xlabel("Number of instances")
    plt.ylabel("Time taken")
    plt.show()

if __name__ =='__main__':
    checkTimeForDiscreteInputDiscreteOutput()'''

def generate_data(N, M, input_type, output_type):
    """
    Generates data based on the input and output type.
    input_type: "discrete" or "real"
    output_type: "discrete" or "real"
    """
    if input_type == "discrete":
        # Generate discrete data (binary features)
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
    else:  # real input
        # Generate real data (continuous features)
        X = pd.DataFrame({i: pd.Series(np.random.randn(N)) for i in range(M)})

    if output_type == "discrete":
        # Generate discrete target (binary)
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    else:  # real output
        # Generate real target (continuous)
        y = pd.Series(np.random.randn(N))
    
    return X, y

def check_time_for_cases():
    cases = [
        ("discrete", "discrete"),
        ("real", "real"),
        ("real", "discrete"),
        ("discrete", "real"),
    ]
    
    # Create subplots: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Varying the number of features M with fixed N
    N = 100  # fixed number of samples
    M_values = range(1, 15)
    
    for input_type, output_type in cases:
        time_for_building_M = []
        time_for_predicting_M = []
        
        for M in M_values:
            # Generate data for varying M
            X, y = generate_data(N, M, input_type, output_type)
            
            # Measure time for fitting
            begin = time.time()
            tree = DecisionTree(criterion="entropy")
            tree.fit(X, y)
            end = time.time()
            time_for_building_M.append(end - begin)
            
            # Measure time for predicting
            begin = time.time()
            y_hat = tree.predict(X)
            end = time.time()
            time_for_predicting_M.append(end - begin)
        
        # Plot the results for building time vs M
        axes[0].plot(M_values, time_for_building_M, label=f"Building - {input_type} input, {output_type} output")
        axes[0].set_title("Time taken to build the tree vs Number of features (M)")
        axes[0].set_xlabel("Number of features (M)")
        axes[0].set_ylabel("Time taken (seconds)")
    
    # Now vary the number of samples N with fixed M
    M = 5  # fixed number of features
    N_values = range(10, 110, 10)
    
    for input_type, output_type in cases:
        time_for_building_N = []
        time_for_predicting_N = []
        
        for N in N_values:
            # Generate data for varying N
            X, y = generate_data(N, M, input_type, output_type)
            
            # Measure time for fitting
            begin = time.time()
            tree = DecisionTree(criterion="entropy")
            tree.fit(X, y)
            end = time.time()
            time_for_building_N.append(end - begin)
            
            # Measure time for predicting
            begin = time.time()
            y_hat = tree.predict(X)
            end = time.time()
            time_for_predicting_N.append(end - begin)
        axes[1].plot(N_values, time_for_building_N, label=f"Building - {input_type} input, {output_type} output")
        axes[1].set_title("Time taken to build the tree vs Number of instances (N)")
        axes[1].set_xlabel("Number of instances (N)")
        axes[1].set_ylabel("Time taken (seconds)")

    # Display legends for both plots
    axes[0].legend()
    axes[1].legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    check_time_for_cases()       
