#!/usr/bin/python3

"""visualizaiton_utils.py

Contains util functions for data and performance visualization."""

__author__ = "Bas Straathof"


import matplotlib.pyplot as plt
import numpy as np


def create_histogram(input_data, title='', rotation=0, fig_size=(12.376, 8),
        xlabel='', ylabel='', normalize=True, bar_labels=False, xticks=False,
        alpha=0.5, rwidth=1, legend=[], save_plot=""):
    """Create a histogram

    Args:
        input_data (list): Input to be plotted
        title       (str): Plot title
        rotation    (int): Degrees of xlabel rotation
        fig_size  (tuple): Size of the image (x-dim, y-dim)
        xlabel      (str): Label of the x-axis
        ylabel      (str): Label of the y-axis
        normalize  (bool): Specifies whether to normalize the data
        bar_labels (bool): Specifies whether to label the bars in the plot
        xticks     (list): List of xticks
        alpha     (float): Tanslucency
        rwidth    (float): The relative width of a bar
        legend     (list): Legend labels
        save_plot   (str): If given, save to this path
    """
    # Create the plot figure
    plt.figure(1, figsize=fig_size)
    # Set the plot title
    if title: plt.title(title, fontsize=20)

    # Histogram of one data set
    if not isinstance(input_data[0], list):
        max_val = int(max(input_data))
        plt.hist(input_data, max_val, density=normalize, alpha=alpha,
                rwidth=rwidth, align='mid')
    # Histogram of multiple data sets
    else:
        # Find the max value over all data sets
        max_val = 0
        for i in input_data:
            max_val = int(max(max(i) + 1, max_val))

        # This only works for two data sets with bars side-by-side
        if rwidth == 0.5:
            plt.hist(input_data, np.arange(0, max_val+1), density=normalize,
                    alpha=alpha, rwidth=rwidth, align='left')
        # This works for multiple data sets with overlapping bars
        else:
            for i in input_data:
                plt.hist(i, np.arange(0, max_val, 5), label=i,
                        density=normalize, alpha=alpha, rwidth=rwidth,
                        align='mid')

    # Set the legend
    if legend: plt.legend(loc='upper right', fontsize='xx-large', labels=legend)

    # Set the x- and y-labels
    if xlabel: plt.xlabel(xlabel, fontsize=22)
    if ylabel: plt.ylabel(ylabel, fontsize=22)

    # Make sure that x-ticks are positioned correctly
    plt.subplots_adjust(bottom=0.3)
    if xticks:
        plt.xticks(np.arange(len(xticks)), xticks, rotation=rotation,
                fontsize=22)

    # Set the size of the x- and y-ticks
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)

    plt.tight_layout()

    # Save the plot
    if save_plot:
        plt.savefig(save_plot, format="pdf", bbox_inches='tight',
                pad_inches=0)
        plt.close()
    else:
        plt.show()
        plt.close()

