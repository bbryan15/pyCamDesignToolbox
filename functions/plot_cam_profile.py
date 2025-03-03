import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_cam_profile(df: pd.DataFrame, title: str = "Cam Profile") -> None:
    """
    Plot cam profile with 4 subplots (s, v, a, j vs ca) in 4 rows and 1 column.
    
    Args:
        df: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
        title: Title for the overall plot
    """
    # Create figure and subplots (4 rows, 1 column)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Plot lift (s) vs cam angle
    ax1.plot(df['ca'], df['s'], 'b-')
    ax1.set_xlabel('Cam Angle (degcm)')
    ax1.set_ylabel('Lift')
    ax1.set_title('Position')
    ax1.grid(True)
    
    # Plot velocity (v) vs cam angle
    ax2.plot(df['ca'], df['v'], 'g-')
    ax2.set_xlabel('Cam Angle (degcm)')
    ax2.set_ylabel('Velocity\n(lift units/degcm)')
    ax2.set_title('Velocity')
    ax2.grid(True)
    
    # Plot acceleration (a) vs cam angle
    ax3.plot(df['ca'], df['a'], 'r-')
    ax3.set_xlabel('Cam Angle (degcm)')
    ax3.set_ylabel('Acceleration\n(lift units/degcm²)')
    ax3.set_title('Acceleration')
    ax3.grid(True)
    
    # Plot jerk (j) vs cam angle
    ax4.plot(df['ca'], df['j'], 'm-')
    ax4.set_xlabel('Cam Angle (degcm)')
    ax4.set_ylabel('Jerk\n(lift units/degcm³)')
    ax4.set_title('Jerk')
    ax4.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show plot
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example data generation (using gcam0_dv_p)
    from gcam0_dv_p import gcam0_dv_p
    
    # Example parameters
    ca1 = 0
    dca = 0.5
    s1 = 0
    s2 = 10
    Vr = 0
    Vmatch = 0.1
    Amatch = 0
    Amx = 0.005
    Dmx = 0.005
    Jmx = 0.0001
    
    # Generate cam profile
    df = gcam0_dv_p(ca1, dca, s1, s2, Vr, Vmatch, Amatch, Amx, Dmx, Jmx)
    
    # Plot the profile
    plot_cam_profile(df, "DV-P Cam Profile")
