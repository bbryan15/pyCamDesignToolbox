import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def plot_cam_profile(dfs: List[pd.DataFrame], labels: List[str]) -> plt.Figure:
    """
    Plot multiple cam profiles with 4 subplots (s, v, a, j vs ca) in 4 rows and 1 column.
    
    Args:
        dfs: List of DataFrames, each with columns ['ca', 's', 'v', 'a', 'j']
        labels: List of labels for each DataFrame
    Returns:
        plt.Figure: The matplotlib figure containing the plots
    """
    # Create figure and subplots (4 rows, 1 column)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot each profile
    for df, label in zip(dfs, labels):
        # Plot lift (s) vs cam angle
        ax1.plot(df['ca'], df['s'], '-', label=label)
        ax1.set_xlabel('Cam Angle')
        ax1.set_ylabel('Lift')
        ax1.set_title('Lift')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        
        # Plot velocity (v) vs cam angle
        ax2.plot(df['ca'], df['v'], '-', label=label)
        ax2.set_xlabel('Cam Angle')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        
        # Plot acceleration (a) vs cam angle
        ax3.plot(df['ca'], df['a'], '-', label=label)
        ax3.set_xlabel('Cam Angle')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration')
        ax3.grid(True)
        ax3.legend(loc='upper right')
        
        # Plot jerk (j) vs cam angle
        ax4.plot(df['ca'], df['j'], '-', label=label)
        ax4.set_xlabel('Cam Angle')
        ax4.set_ylabel('Jerk')
        ax4.set_title('Jerk')
        ax4.grid(True)
        ax4.legend(loc='upper right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig