import numpy as np
import pandas as pd

def rocr(csvajr: pd.DataFrame, Rbc: float, Rrlr: float) -> np.ndarray:
    """
    Computes cam radius of curvature from reciprocating follower lift.
    
    Args:
        csvajr: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: reciprocating follower cam angle vector (degcm)
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2)
            j: reciprocating follower jerk vector (mm/degcm^3) NOT USED
        Rbc: cam base circle radius (mm)
        Rrlr: roller follower radius (mm)
    
    Returns:
        numpy.ndarray: ROC - cam radius of curvature vector (mm)
    """
    # Extract input vectors
    Lft = csvajr['s'].values
    Vel = csvajr['v'].values
    Acl = csvajr['a'].values
    
    # Calculate distance from cam axis to roller axis
    R2rlr = Lft + Rbc + Rrlr
    
    # Calculate radius of curvature
    ROC = (
        (R2rlr**2 + (Vel*180/np.pi)**2)**1.5 / 
        (R2rlr**2 + 2*(Vel*180/np.pi)**2 - R2rlr*Acl*(180/np.pi)**2)
    ) - Rrlr
    
    return ROC
