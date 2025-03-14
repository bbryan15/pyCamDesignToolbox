import numpy as np
import pandas as pd

def of2vlv(csvajo: pd.DataFrame, Arcrbd: float, Arlr2bald: float, 
           Avlvd: float, Rrcr2bal: float) -> tuple[pd.DataFrame, float]:
    """
    Converts from oscillating follower to valve.
    
    Args:
        csvajo: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: oscillating rocker follower cam angle vector (degcm)
            s: oscillating rocker follower angle vector, =0 on base circle (deg)
            v: oscillating rocker follower angular velocity vector (deg/degcm)
            a: oscillating rocker follower angular acceleration vector (deg/degcm^2)
            j: oscillating rocker follower angular vector (deg/degcm^3)
        Arcrbd: angle of line from rocker axis to roller axis on base circle (deg)
        Arlr2bald: angle between lines from rocker axis to roller axis and 
                  e-foot ball center over the top of the rocker (deg)
        Avlvd: angle of normal to valve axis relative to horizontal X axis (deg)
        Rrcr2bal: dist from rocker axis to e-foot ball axis (mm)
    
    Returns:
        tuple[pd.DataFrame, float]: 
            - DataFrame with columns ['ca', 's', 'v', 'a', 'j']
                ca: oscillating rocker follower cam angle vector (degcm)
                s: valve lift vector (mm)
                v: valve velocity vector (mm/degcm)
                a: valve acceleration vector (mm/degcm^2)
                j: valve jerk vector (mm/degcm^3)
            - Yrcr2balb: dist parallel to valve axis from rocker axis to 
                        e-foot ball center on cam base circle (mm)
    """
    # Extract input vectors
    Acmpd = csvajo['ca'].values
    dArcrd = csvajo['s'].values
    
    # Calculate angles in radians
    Arcr = (Arcrbd - dArcrd) * np.pi/180  # angle of line from rocker axis to roller axis
    Arcr2bal = Arcr - Arlr2bald * np.pi/180  # angle line from rocker axis to e-foot ball center
    Arcr2balv = Arcr2bal - Avlvd * np.pi/180  # angle between normal to valve axis and line
    
    # Calculate distances
    Yrcr2bal = Rrcr2bal * np.sin(Arcr2balv)  # distance parallel to valve axis
    Xv = Yrcr2bal[0] - Yrcr2bal  # valve lift
    
    # Calculate exact derivatives
    # Valve velocity
    Vv = Rrcr2bal * np.cos(Arcr2balv) * np.pi/180 * csvajo['v'].values
    
    # Valve acceleration
    Av = Rrcr2bal * (
        np.cos(Arcr2balv) * np.pi/180 * csvajo['a'].values +
        np.sin(Arcr2balv) * (np.pi/180 * csvajo['v'].values)**2
    )
    
    # Valve jerk
    Jv = Rrcr2bal * (
        np.cos(Arcr2balv) * np.pi/180 * csvajo['j'].values +
        3 * np.sin(Arcr2balv) * (np.pi/180)**2 * csvajo['v'].values * csvajo['a'].values -
        np.cos(Arcr2balv) * (np.pi/180 * csvajo['v'].values)**3
    )
    
    # Create output DataFrame
    csvajv = pd.DataFrame({
        'ca': Acmpd,
        's': Xv,
        'v': Vv,
        'a': Av,
        'j': Jv
    })
    
    # Return both DataFrame and base circle distance
    return csvajv, Yrcr2bal[0] 