import numpy as np
import pandas as pd

def vlv2of(csvajv: pd.DataFrame, Arlr2bald: float, Avlvd: float, 
           Rrcr2bal: float, Yrcr2balb: float) -> tuple[pd.DataFrame, float]:
    """
    Converts from valve to oscillating follower.
    
    Args:
        csvajv: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: oscillating rocker follower cam angle vector (degcm)
            s: valve lift vector (mm)
            v: valve velocity vector (mm/degcm)
            a: valve acceleration vector (mm/degcm^2)
            j: valve jerk vector (mm/degcm^3)
        Arlr2bald: angle between lines from rocker axis to roller axis and 
                  e-foot ball center over the top of the rocker (deg)
        Avlvd: angle of normal to valve axis relative to horizontal X axis (deg)
        Rrcr2bal: dist from rocker axis to e-foot ball axis (mm)
        Yrcr2balb: dist parallel to valve axis from rocker axis to 
                  e-foot ball center on cam base circle (mm)
    
    Returns:
        tuple[pd.DataFrame, float]:
            - DataFrame with columns ['ca', 's', 'v', 'a', 'j']
                ca: oscillating rocker follower cam angle vector (degcm)
                s: oscillating rocker follower angle vector, =0 on base circle (deg)
                v: oscillating rocker follower angular velocity vector (deg/degcm)
                a: oscillating rocker follower angular acceleration vector (deg/degcm^2)
                j: oscillating rocker follower angular jerk vector (deg/degcm^3)
            - Arcrbd: angle of line from rocker axis to roller axis on base circle (deg)
    """
    # Extract input vectors
    Acmpd = csvajv['ca'].values
    Xv = csvajv['s'].values
    
    # Calculate distances and angles
    Yrcr2bal = Yrcr2balb - Xv  # dist parallel to valve axis
    Arcr2balv = np.arcsin(Yrcr2bal/Rrcr2bal)  # angle between normal to valve axis and line
    Arcr2bal = Arcr2balv + Avlvd * np.pi/180  # angle line from rocker axis
    Arcr = Arcr2bal + Arlr2bald * np.pi/180  # angle of line from rocker axis to roller axis
    dArcr = Arcr[0] - Arcr  # rocker lift angle, =0 on base circle
    dArcrd = dArcr * 180/np.pi
    
    # Calculate derivatives
    u = Yrcr2bal/Rrcr2bal
    sqrt_term = np.sqrt(1 - u**2)
    
    # Angular velocity
    Avel = 180/np.pi * (1/sqrt_term) * csvajv['v'].values/Rrcr2bal
    
    # Angular acceleration
    Aacl = 180/np.pi * (
        (1/sqrt_term) * csvajv['a'].values/Rrcr2bal -
        u/(sqrt_term**3) * (csvajv['v'].values/Rrcr2bal)**2
    )
    
    # Angular jerk
    Ajrk = 180/np.pi * (
        (1/sqrt_term) * csvajv['j'].values/Rrcr2bal -
        3*u/(sqrt_term**3) * csvajv['v'].values * csvajv['a'].values/Rrcr2bal**2 +
        ((1/(sqrt_term**3)) - 2*u**2/(sqrt_term**5)) * (csvajv['v'].values/Rrcr2bal)**3
    )
    
    # Create output DataFrame
    csvajo = pd.DataFrame({
        'ca': Acmpd,
        's': dArcrd,
        'v': Avel,
        'a': Aacl,
        'j': Ajrk
    })
    
    # Calculate base circle angle
    Arcrbd = Arcr[0] * 180/np.pi
    
    return csvajo, Arcrbd 