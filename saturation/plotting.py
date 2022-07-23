import io

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from saturation.crater_record import CraterRecord
from saturation.geometry import Location, Arc


def plot_circle(center: Location,
                radius: float,
                axes_subplot,
                fill: bool = False,
                color: str = 'black',
                lw: float = 1,
                antialiased: bool = True):
    """
    Plots the specified circle on the supplied subplot.
    """
    axes_subplot.add_patch(matplotlib.patches.Circle(center,
                                                     radius=radius,
                                                     color=color,
                                                     fill=fill,
                                                     lw=lw,
                                                     antialiased=antialiased))


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


def plot_crater_record(crater_record: CraterRecord,
                       observed_terrain_size: float,
                       terrain_padding: float,
                       figsize: float = 4):
    """
    Plots all craters in the crater record.
    Erased rim arcs are shown in blue.
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    ax.set_xlim([terrain_padding, observed_terrain_size + 1])
    ax.set_ylim([terrain_padding, observed_terrain_size + 1])

    # Plot craters
    for crater in crater_record.all_craters:
        plot_circle((crater.x, crater.y), crater.radius, ax)

    # Plot erased rim arcs
    for crater_id, arcs in crater_record._erased_arcs.items():
        crater = crater_record.get_crater(crater_id)

        for arc in arcs:
            plot_arc((crater.x, crater.y),
                     crater.radius,
                     (arc[0], arc[1]),
                     ax,
                     color='blue',
                     lw=2)

    plt.show()


def convert_plot_to_array(fig, show_plot: bool = False) -> np.array:
    """
    Converts a plot to a 2D binary numpy array.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='raw')

    if not show_plot:
        plt.close()

    buffer.seek(0)
    img = np.reshape(np.frombuffer(buffer.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    squashed = img.sum(axis=2)

    return np.where(squashed > 764, 0, 1)