import numpy as np
import pandas as pd

def cs2rf(csvc: pd.DataFrame, Rbc: float, Rrlr: float, rot: int) -> pd.DataFrame:
    """
    Converts from cam surface to reciprocating follower.
    
    Args:
        csvc: DataFrame with columns ['Atand', 'Rtan', 'dRtan']
            Atand: cam profile angle vector (angle counterclockwise from cam profile 
                  reference to line from cam axis to point of roller tangent contact, degcm)
            Rtan: cam radius at the point of tangent contact between cam and roller vector (mm)
            dRtan: dRtan/dAtand (mm/deg)
        Rbc: cam base circle radius (mm)
        Rrlr: roller follower radius (mm)
        rot: +1 clockwise, -1 counter-clockwise, looking at the cam with the valve 
             on the right or initial rocker lift clockwise
    
    Returns:
        pd.DataFrame: DataFrame with columns ['ca', 's']
            ca: reciprocating follower cam angle vector (increasing with crank angle, degcm)
            s: reciprocating follower lift vector (mm)
    """
    # Convert angles to radians and extract vectors
    Atan = np.pi/180 * csvc['Atand'].values
    Rtan = csvc['Rtan'].values
    dRtan = 180/np.pi * csvc['dRtan'].values
    
    # Initialize arrays for results
    nr = len(Atan)
    Acm = np.zeros(nr)
    Lft = np.zeros(nr)
    
    # Calculate reciprocating follower lift vs. cam rotational angle using vector algebra
    for i in range(nr):
        # Rtan vector from cam axis to point of roller contact
        XYtan = np.array([
            Rtan[i] * np.cos(Atan[i]),
            Rtan[i] * np.sin(Atan[i])
        ])
        
        # dRtan/dAtan vector
        dXYtan = np.array([
            -Rtan[i] * np.sin(Atan[i]) + dRtan[i] * np.cos(Atan[i]),
            Rtan[i] * np.cos(Atan[i]) + dRtan[i] * np.sin(Atan[i])
        ])
        
        # Angle of dRtan/dAtan vector tangent to cam surface
        AdXYtan = np.arctan2(dXYtan[1], dXYtan[0])
        
        # Vector from roller axis to point of cam contact
        XYrlr = np.array([
            Rrlr * np.cos(AdXYtan + np.pi/2),
            Rrlr * np.sin(AdXYtan + np.pi/2)
        ])
        
        # R2rlr vector from cam axis to pitch curve
        XY2rlr = XYtan - XYrlr
        
        # Convert to polar coordinates
        Acm[i] = np.arctan2(XY2rlr[1], XY2rlr[0])
        R2rlr = np.sqrt(XY2rlr[0]**2 + XY2rlr[1]**2)
        
        # Calculate lift
        Lft[i] = R2rlr - Rbc - Rrlr
    
    # Sign change to properly render cam surface
    Acm = rot * Acm
    
    # Wrap Acm to rot*Atan[0] to preserve wrapping angle
    Acm = np.mod(Acm - rot*Atan[0], 2*np.pi) + rot*Atan[0]
    
    # Convert angles to degrees
    Acmd = Acm * 180/np.pi
    
    # Create output DataFrame
    return pd.DataFrame({
        'ca': Acmd,
        's': Lft
    }) 