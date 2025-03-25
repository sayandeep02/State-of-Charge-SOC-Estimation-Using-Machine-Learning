import numpy as np
from sklearn import metrics
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.cm as cm
import squarify

def normalize(X):
    """
    Normalize the input data matrix `X` using min-max scaling to a range of [-1, 1].

    This function performs normalization by scaling each feature of the data matrix 
    `X` to a range between -1 and 1. The normalization formula used is:
    ((2*(X[i][j] - min_value)) / (max_value - min_value)) - 1, where `min_value` 
    and `max_value` are the minimum and maximum values of the feature in each row.

    Parameters:
    X (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.

    Returns:
    numpy.ndarray: A 2D numpy array of the same shape as `X` with normalized values.
    """
    normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(X.shape[1]):
            normalized_X[i][j] = ((2*(X[i][j]-min_value))/(max_value-min_value))-1
    return normalized_X


def get_metrics(y_test, y_predicted):
    """
    Calculate various performance metrics and return them in a styled DataFrame.

    Computes the following metrics: max error, mean absolute error, mean absolute percentage error,
    mean squared error, root mean squared error, and root mean squared log error.

    Parameters:
    y_test (numpy.ndarray): Array of true target values.
    y_predicted (numpy.ndarray): Array of predicted target values.

    Returns:
    tuple: A tuple containing a list of metric values and a styled DataFrame of the metrics.
    """
    ma = metrics.max_error(y_test, y_predicted)
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mape = metrics.mean_absolute_percentage_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    rmse = metrics.root_mean_squared_error(y_test, y_predicted)
    rmsle = metrics.root_mean_squared_log_error(y_test, y_predicted)
    results = {
        "Metric": [
            "max_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "root_mean_squared_log_error"
        ],
        "Value": [
            ma,
            mae,
            mape,
            mse,
            rmse,
            rmsle
        ]
    }
    df = pd.DataFrame(results)
    styled_table = df.style.background_gradient(cmap="coolwarm").set_properties(**{
        'border': '1.5px solid black', 
        'color': 'black',
        'font-size': '12pt',
        'text-align': 'center'
    })
    return [ma, mae, mape, mse, rmse, rmsle], styled_table
    

def metrics_plot(models_names, metrics_names, metrics_values):
    """
    Create a bar plot to visualize metrics for different models.

    Parameters:
    models_names (list of str): Names of the models.
    metrics_names (list of str): Names of the metrics to plot.
    metrics_values (list of lists): Metrics values for each model. Each sublist corresponds to a metric.

    Raises:
    ValueError: If the number of metric names does not match the number of metric values.
    """
    if len(metrics_names) != len(metrics_values[0]):
        raise ValueError("len(metrics_names) is not equal to len(metrics_values).")
    metrics_grouped = {label: [metric[i] * 100 for metric in metrics_values] for i, label in enumerate(metrics_names)}
    x = np.arange(len(models_names))
    width = 0.1
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(14, 4))
    for attribute, measurement in metrics_grouped.items():
        offset = width * multiplier
        rects = ax.bar(x+offset, measurement, width, label=attribute)
        ax.bar_label(rects, fmt='%.1f', padding=3)
        multiplier += 1
    ax.set_ylabel('[%]')
    ax.set_title('Metrics by Models')
    ax.set_xticks(x + width * (len(metrics_grouped) - 1) / 2)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    plt.show()
        

def results_plot(
    x, 
    y_observed,
    y_predicted, 
    y_predicted_ahif,
    xlim,
    ylim,
    xlabel, 
    ylabel, 
    title,
    color_observed='#3643f5',
    color_predicted='#f53333',
    color_predicted_ahif='#13ad2d'
):
    """
    Create plots to visualize the observed and predicted values, including predictions from an alternative model (AHIF).

    Parameters:
    x (numpy.ndarray): The x-axis values (e.g., time steps).
    y_observed (numpy.ndarray): The observed values.
    y_predicted (numpy.ndarray): The predicted values.
    y_predicted_ahif (numpy.ndarray): The predicted values from the alternative model (AHIF).
    xlim (tuple): x-axis limits as (min, max).
    ylim (tuple): y-axis limits as (min, max).
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    color_observed (str): Color for the observed data.
    color_predicted (str): Color for the predicted data.
    color_predicted_ahif (str): Color for the predicted data from the alternative model.
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11, 6))
    observed_handle = mlines.Line2D([], [], color=color_observed, label='observed')
    predicted_handle = mlines.Line2D([], [], color=color_predicted, marker='o', markersize=5, linestyle='None', label='predicted')
    predicted_ahif_handle = mlines.Line2D([], [], color=color_predicted_ahif, label='predicted-AHIF')
    
    axs[0].plot(x, y_observed, label='observed', color=color_observed)
    axs[0].scatter(x, y_predicted, label='predicted', s=0.005, color=color_predicted)
    axs[0].plot(x, y_predicted_ahif, label='predicted-AHIF', color=color_predicted_ahif)
    axs[0].set_xlabel(f'{xlabel}'), axs[0].set_ylabel(f'{ylabel}')
    axs[0].legend(handles=[observed_handle, predicted_handle, predicted_ahif_handle])
    
    axs[1].plot(x, y_observed, label='observed', color=color_observed)
    axs[1].scatter(x, y_predicted, label='predicted', s=0.005, color=color_predicted)
    axs[1].plot(x, y_predicted_ahif, label='predicted-AHIF', color=color_predicted_ahif)
    axs[1].set_xlabel(f'{xlabel}'), axs[1].set_ylabel(f'{ylabel}')
    axs[1].legend(handles=[observed_handle, predicted_handle, predicted_ahif_handle])
    axs[1].set_xlim(xlim[0], xlim[1])
    axs[1].set_ylim(ylim[0], ylim[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def pie_chart_plot(train_data_df, val_data_df, test_data_df, title="Dataset Distribution"):
    """
    Create a pie chart to visualize the distribution of the dataset among training, validation, and test sets.

    Parameters:
    train_data_df (pd.DataFrame): DataFrame containing training data.
    val_data_df (pd.DataFrame): DataFrame containing validation data.
    test_data_df (pd.DataFrame): DataFrame containing test data.
    title (str): Title of the pie chart.
    """
    num_rows_train = len(train_data_df)
    num_rows_val = len(val_data_df)
    num_rows_test = len(test_data_df)

    total_rows = num_rows_train + num_rows_val + num_rows_test

    percentages = [
        num_rows_train / total_rows * 100,
        num_rows_val / total_rows * 100,
        num_rows_test / total_rows * 100
    ]

    labels = ['Training', 'Validation', 'Test']
    
    plt.pie(percentages, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.axis('equal')
    plt.show()


def violin_plot(train_data_df, val_data_df, test_data_df):
    """
    Create violin plots to visualize the distribution of features for training, validation, and test datasets.

    Parameters:
    train_data_df (pd.DataFrame): DataFrame containing training data.
    val_data_df (pd.DataFrame): DataFrame containing validation data.
    test_data_df (pd.DataFrame): DataFrame containing test data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    sns.violinplot(ax=axes[0], data=train_data_df)
    axes[0].set_title('Training Distribution')
    sns.violinplot(ax=axes[1], data=val_data_df)
    axes[1].set_title('Validation Distribution')
    sns.violinplot(ax=axes[2], data=test_data_df)
    axes[2].set_title('Testing Distribution')
    

def hist_plot(train_data_df, val_data_df, test_data_df):
    """
    Create histograms to visualize the distribution of each feature for training, validation, and test datasets.

    Parameters:
    train_data_df (pd.DataFrame): DataFrame containing training data.
    val_data_df (pd.DataFrame): DataFrame containing validation data.
    test_data_df (pd.DataFrame): DataFrame containing test data.
    """
    fig, axs = plt.subplots(1, train_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(train_data_df.keys()):
        sns.histplot(ax=axs[i], x=train_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Training Distribution', fontsize=16, y=1.06)
    plt.show()
    
    fig, axs = plt.subplots(1, val_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(val_data_df.keys()):
        sns.histplot(ax=axs[i], x=val_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Validation Distribution', fontsize=16, y=1.06)
    plt.show()
    
    fig, axs = plt.subplots(1, test_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(test_data_df.keys()):
        sns.histplot(ax=axs[i], x=test_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Testing Distribution', fontsize=16, y=1.06)
    plt.show()


def time_series_plot(train_data_df, val_data_df, test_data_df):
    """
    Create line plots to visualize time series data for training, validation, and test datasets.

    Parameters:
    train_data_df (pd.DataFrame): DataFrame containing training data.
    val_data_df (pd.DataFrame): DataFrame containing validation data.
    test_data_df (pd.DataFrame): DataFrame containing test data.
    """
    fig, axs = plt.subplots(1, train_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(train_data_df.keys()):
        sns.lineplot(ax=axs[i], data=train_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Training Distribution', fontsize=16, y=1.06)
    plt.show()
    
    fig, axs = plt.subplots(1, val_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(val_data_df.keys()):
        sns.lineplot(ax=axs[i], data=val_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Validation Distribution', fontsize=16, y=1.06)
    plt.show()
    
    fig, axs = plt.subplots(1, test_data_df.shape[1], figsize=(20, 4))
    for i, key in enumerate(test_data_df.keys()):
        sns.lineplot(ax=axs[i], data=test_data_df[key])
        axs[i].legend([f'{key}'])
    plt.tight_layout()
    plt.suptitle('Testing Distribution', fontsize=16, y=1.06)
    plt.show()


def correlation_map(data_df):
    """
    Create a heatmap to visualize the correlation matrix of the features in the dataset.

    Parameters:
    data_df (pd.DataFrame): DataFrame containing the dataset.
    """
    plt.figure(figsize=(16, 6))
    sns.heatmap(data_df.corr(), annot=True)
    plt.show()
    




def treemap_plot(train_data_df, val_data_df, test_data_df, title="Dataset Distribution"):
    """
    Create a TreeMap to visualize the distribution of the dataset among training, validation, and test sets.

    Parameters:
    train_data_df (pd.DataFrame): DataFrame containing training data.
    val_data_df (pd.DataFrame): DataFrame containing validation data.
    test_data_df (pd.DataFrame): DataFrame containing test data.
    title (str): Title of the TreeMap.
    """
    # Get row counts
    num_rows_train = len(train_data_df)
    num_rows_val = len(val_data_df)
    num_rows_test = len(test_data_df)

    # Calculate percentages
    total_rows = num_rows_train + num_rows_val + num_rows_test
    sizes = [
        num_rows_train / total_rows * 100,
        num_rows_val / total_rows * 100,
        num_rows_test / total_rows * 100
    ]

    # Labels with percentages
    labels = [
        f'Training\n{sizes[0]:.1f}%',
        f'Validation\n{sizes[1]:.1f}%',
        f'Test\n{sizes[2]:.1f}%'
    ]

    # Colors
    # colors = ['#8e44ad', '#1abc9c', '#e74c3c']  # Purple, Teal, Coral
    colors = cm.viridis(np.linspace(0, 1, len(sizes)))

    # Plot TreeMap
    fig, ax = plt.subplots(figsize=(12, 7))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize': 14})
    
    plt.title(title, fontsize=20, fontweight='bold', color='#34495E')
    plt.axis('off')  # Remove axis lines
    plt.show()