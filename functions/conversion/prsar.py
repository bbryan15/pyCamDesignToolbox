import numpy as np
import pandas as pd

def prsar(csvajr: pd.DataFrame, Rbc: float, Rrlr: float) -> np.ndarray:
    """
    Computes reciprocating follower pressure angle.
    
    The pressure angle is the angle between the direction of cam-roller force 
    and the instantaneous direction of roller motion, which is in the direction 
    of a line between the cam shaft axis and the roller axis.
    
    Args:
        csvajr: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: reciprocating follower cam angle vector (degcm) NOT USED
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2) NOT USED
            j: reciprocating follower jerk vector (mm/degcm^3) NOT USED
        Rbc: cam base circle radius (mm)
        Rrlr: roller follower radius (mm)
    
    Returns:
        numpy.ndarray: Aprsrd - reciprocating follower pressure angle vector (deg)
                      positive for increasing cam lift
    """
    # Extract required vectors
    Lft = csvajr['s'].values
    Vel = csvajr['v'].values
    
    # Calculate distance from cam axis to roller axis
    R2rlr = Lft + Rbc + Rrlr
    
    # Calculate pressure angle in radians
    Aprsr = np.arctan(Vel * 180/np.pi / R2rlr)
    
    # Convert to degrees
    Aprsrd = Aprsr * 180/np.pi
    
    return Aprsrd 