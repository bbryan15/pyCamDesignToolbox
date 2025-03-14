import numpy as np
import pandas as pd

def ofrr(dArcrd: np.ndarray, Arcrbd: float, Arlr2bald: float, Avlvd: float, 
         Rrcr2bal: float, Rrcr2rlr: float) -> np.ndarray:
    """
    Calculates rocker ratio for oscillating follower.
    
    The rocker ratio is the derivative of valve lift with respect to instantaneous 
    roller displacement, which is perpendicular to a line between the rockershaft 
    axis and the roller axis.
    
    Args:
        dArcrd: rocker lift angle vector, =0 on base circle (deg)
        Arcrbd: angle of line from rocker axis to roller axis on base circle (deg)
        Arlr2bald: angle between lines from rocker axis to roller axis and 
                  e-foot ball center over the top of the rocker (deg)
        Avlvd: angle of normal to valve axis relative to horizontal X axis (deg)
        Rrcr2bal: distance from rocker axis to e-foot ball axis (mm)
        Rrcr2rlr: distance from rocker axis to roller axis (mm)
    
    Returns:
        numpy.ndarray: rkrrat - ratio of valve lift to lift normal to 
        rockershaft-roller moment arm = ratio of force normal to 
        rockershaft-roller moment arm to valve force
    """
    # Calculate angles in radians
    Arcr2bal = (Arcrbd - Arlr2bald - dArcrd) * np.pi/180
    Arcr2balv = Arcr2bal - Avlvd * np.pi/180
    
    # Calculate rocker ratio
    rkrrat = Rrcr2bal * np.cos(Arcr2balv) / Rrcr2rlr
    
    return rkrrat 