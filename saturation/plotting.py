import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from saturation.geometry import Location, Arc


def plot_circle(center: Location,
                radius: float,
                axes_subplot,
                fill: bool = False,
                color: str = 'black',
                lw: float = 1):
    """
    Plots the specified circle on the supplied subplot.
    """
    axes_subplot.add_patch(matplotlib.patches.Circle(center,
                                                     radius=radius,
                                                     color=color,
                                                     fill=fill,
                                                     lw=lw))


def plot_arc(center: Location,
             radius: float,
             arc: Arc,
             axes_subplot,
             color: str = 'black',
             lw: float = 1):
    """
    Plots the specified arc on the supplied subplot.
    """
    axes_subplot.add_patch(matplotlib.patches.Arc(center,
                                                  width=radius * 2,
                                                  height=radius * 2,
                                                  theta1=np.rad2deg(arc[0]),
                                                  theta2=np.rad2deg(arc[1]),
                                                  color=color,
                                                  lw=lw))


def plot_up_to_crater(crater_id: int,
                      craters: pd.DataFrame,
                      erased_rim_arcs: pd.DataFrame,
                      scale: float,
                      figsize: float = 4):
    """
    Plots all craters up to the specified crater id.
    Erased rim arcs are shown in blue.
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    ax.set_xlim([0, scale])
    ax.set_ylim([0, scale])

    # Plot craters
    for row in craters.loc[range(1, crater_id + 1)].itertuples():
        plot_circle((row.x, row.y), row.radius, ax)

    # Plot erased rim arcs
    filtered_erased_rim_arcs = erased_rim_arcs[erased_rim_arcs.new_id <= crater_id]
    for row in filtered_erased_rim_arcs.itertuples():
        old_crater = craters.loc[row.old_id]

        plot_arc((old_crater.x, old_crater.y),
                 old_crater.radius,
                 (row.theta1, row.theta2),
                 ax,
                 color='blue',
                 lw=2)

    plt.show()
