import numpy as np
import pandas as pd

def prsao(csvajr: pd.DataFrame, Rbc: float, Rrcr2rlr: float, Rrlr: float, 
          Xrcr: float, Yrcr: float) -> np.ndarray:
    """
    Computes oscillating follower pressure angle.
    
    The pressure angle is the angle between the direction of cam-roller force 
    and the instantaneous direction of roller motion, which is perpendicular 
    to a line between the rockershaft axis and the roller axis.
    
    Args:
        csvajr: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: reciprocating follower cam angle vector (degcm) NOT USED
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2) NOT USED
            j: reciprocating follower jerk vector (mm/degcm^3) NOT USED
        Rbc: cam base circle radius (mm)
        Rrcr2rlr: distance from rocker axis to roller axis (mm)
        Rrlr: roller follower radius (mm)
        Xrcr: horizontal distance from cam axis to rocker axis (mm)
        Yrcr: vertical distance from cam axis to rocker axis (mm)
    
    Returns:
        numpy.ndarray: Aprsod - oscillating follower pressure angle vector (deg)
    """
    # Extract required vectors
    Lft = csvajr['s'].values
    Vel = csvajr['v'].values
    
    # Calculate distances and angles
    R2rlr = Lft + Rbc + Rrlr  # distance from cam axis to roller axis (mm)
    Aprsr = np.arctan(Vel * 180/np.pi / R2rlr)  # reciprocating follower pressure angle (rad)
    R2rcr = np.sqrt(Xrcr**2 + Yrcr**2)  # distance from cam axis to rocker axis (mm)
    A2rcr = np.arctan2(Yrcr, Xrcr)  # angle of line from cam axis to rocker axis (rad)
    
    # Calculate angle of line from cam axis to roller axis
    cos_term = (R2rlr**2 + R2rcr**2 - Rrcr2rlr**2)/(2*R2rlr*R2rcr)
    A2rlr = A2rcr + np.arccos(cos_term)
    
    # Calculate angle of line from rocker axis to roller axis
    Arcr = np.arctan2(R2rlr*np.sin(A2rlr) - Yrcr,
                     R2rlr*np.cos(A2rlr) - Xrcr)
    
    # Convert all angles to positive (0 to 2π)
    Arcr = np.mod(Arcr, 2*np.pi)
    
    # Calculate oscillating follower pressure angle
    Aprso = Aprsr + Arcr - 0.5*np.pi - A2rlr
    
    # Convert to degrees
    Aprsod = Aprso * 180/np.pi
    
    return Aprsod 