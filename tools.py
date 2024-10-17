import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def calculate_performance_stats(data, param_cols):
    """
    Calculate the mean and standard deviation of performance for each unique combination of the given parameters.

    Parameters:
    - data (np.ndarray): A 2D numpy array where:
        - Column 0: Dataset index
        - Column 1: Run
        - Column N: Params (param_1, param_2, etc.)
        - Last Column: Performance
    - param_cols (list of int): List of column indices representing the parameters.

    Returns:
    - performance_stats_array (np.ndarray): A 2D array containing unique combinations of parameters, mean performance, and standard deviation, formatted to 3 decimal places.
    - best_combination (list): The combination of parameters with the best performance.
    """
    
    # Extract the parameter columns and the performance column
    params = data[:, param_cols]  # Select all parameter columns
    performance = data[:, -1]  # Assuming the performance is in the last column

    # Find unique combinations of the parameters
    unique_params = np.unique(params, axis=0)

    # Initialize a list to hold the mean performance and std deviation for each combination
    performance_stats = []

    # Calculate the mean performance and standard deviation for each unique combination
    for param_set in unique_params:
        # Get the indices where all parameters match the current combination
        mask = np.all(params == param_set, axis=1)
        mean_perf = np.mean(performance[mask])  # Calculate the mean performance
        std_perf = np.std(performance[mask])    # Calculate the standard deviation of performance
        performance_stats.append(list(param_set) + [mean_perf, std_perf])

    # Convert the result to a numpy array
    performance_stats_array = np.array(performance_stats)

    # Format the performance statistics to 3 decimal places
    formatted_performance_stats_array = np.array([[*params[:-2], f"{params[-2]:.3f}", f"{params[-1]:.3f}"]
                                                   for params in performance_stats_array])

    # Find the combination with the best performance (highest mean)
    best_index = np.argmax(performance_stats_array[:, -2])  # Index of the best performance
    best_combination = performance_stats_array[best_index]  # Get the best combination

    return formatted_performance_stats_array, best_combination

def plot_results(nb_hidden_layers, reg_interest, best_y_test, best_predicted_classes, best_loss_vs_epoch, compendium, l_rate):
    # Generate the confusion matrix
    cm = confusion_matrix(best_y_test, best_predicted_classes.numpy())

    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=reg_interest, yticklabels=reg_interest)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'images/MLP/MLP_{nb_hidden_layers}_ConfusionMatrix.png')  # Save the figure
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()

    param_index = ((np.linspace(0, nb_hidden_layers-1,nb_hidden_layers)+2).astype(int)).tolist()
    performance_stats_result, best_combination_result = calculate_performance_stats(compendium, param_index)

    print("Best Combination:")
    curve_label = ""
    for i in range(len(param_index)):
        print(f'nb_hn_l{i}: {int(best_combination_result[i])}')    
        curve_label += f"HL{i+1}: {best_combination_result[i]} HN   "
    print(f'Mean(Acc): {float(best_combination_result[nb_hidden_layers]):.3f}')
    print(f'Stdev(Acc): {float(best_combination_result[nb_hidden_layers+1]):.3f}')


    plt.figure(figsize=(6,4))

    plt.scatter(range(len(best_loss_vs_epoch)), best_loss_vs_epoch, color='g', label=curve_label, marker='o')

    # Add title and labels
    plt.title('Best Loss vs Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.xlim([0, 500])
    plt.ylim([0, 2])
    plt.ylabel('Best Loss', fontsize=12)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend
    plt.legend()

    # Get the current date and time
    now = datetime.now()
    timestamp = now.strftime('%y%m%d_%H_%M')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'images/MLP/MLP_{nb_hidden_layers}_LearningCurve_{l_rate}_{timestamp}.png')  # Save the figure
    plt.savefig(f'images/MLP/MLP_{nb_hidden_layers}_LearningCurve.png')  # Save the figure
    plt.show()
    plt.close()