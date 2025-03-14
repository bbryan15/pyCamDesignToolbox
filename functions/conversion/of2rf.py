import numpy as np
import pandas as pd

def of2rf(csvajo: pd.DataFrame, Arcrbd: float, Rbc: float, Rrcr2rlr: float, 
          Rrlr: float, Xrcr: float, Yrcr: float, rot: int) -> pd.DataFrame:
    """
    Converts from oscillating follower to reciprocating follower.
    
    Args:
        csvajo: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: cam angle vector (degcm)
            s: oscillating rocker follower angle vector, =0 on base circle (deg)
            v: oscillating rocker follower angular velocity vector (deg/degcm)
            a: oscillating rocker follower angular acceleration vector (deg/degcm^2)
            j: oscillating rocker follower angular vector (deg/degcm^3)
        Arcrbd: angle of line from rocker axis to roller axis on base circle (deg)
        Rbc: cam base circle radius (mm)
        Rrcr2rlr: distance from rocker axis to roller axis (mm)
        Rrlr: roller follower radius (mm)
        Xrcr: horizontal distance from cam axis to rocker axis (mm)
        Yrcr: vertical distance from cam axis to rocker axis (mm)
        rot: +1 clockwise, -1 counter-clockwise
    
    Returns:
        DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: cam angle vector (degcm)
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2)
            j: reciprocating follower jerk vector (mm/degcm^3)
    """
    # Extract input vectors
    Acmpd = csvajo['ca'].values
    dArcrd = csvajo['s'].values
    
    # Calculate angles and positions
    Arcr = (Arcrbd - dArcrd) * np.pi/180  # Convert to radians
    Arcrb = Arcrbd * np.pi/180
    X2rlr = Xrcr + Rrcr2rlr * np.cos(Arcr)
    Y2rlr = Yrcr + Rrcr2rlr * np.sin(Arcr)
    R2rlr = np.sqrt(X2rlr**2 + Y2rlr**2)
    Lft = R2rlr - Rbc - Rrlr
    A2rlr = np.arctan2(Y2rlr, X2rlr)
    A2rlrb = np.arctan2(Yrcr + Rrcr2rlr * np.sin(Arcrb), 
                        Xrcr + Rrcr2rlr * np.cos(Arcrb))
    Acmd = Acmpd - 180/np.pi * rot * (A2rlrb - A2rlr)
    
    # Calculate first derivatives
    dArcrdAcmp = -csvajo['v'].values
    dLftdAcmp = Rrcr2rlr * (-X2rlr * np.sin(Arcr) + Y2rlr * np.cos(Arcr)) * dArcrdAcmp / R2rlr
    u2 = Y2rlr / X2rlr
    du2dAcmp = Rrcr2rlr * (np.cos(Arcr) + u2 * np.sin(Arcr)) * dArcrdAcmp / X2rlr
    dA2rlrdAcmp = du2dAcmp / (1 + u2**2)
    dAcmdAcmp = 1 + rot * dA2rlrdAcmp
    Vel = np.pi/180 * dLftdAcmp / dAcmdAcmp
    
    # Calculate second derivatives
    d2ArcrdAcmp2 = -csvajo['a'].values * 180/np.pi
    d2LftdAcmp2 = Rrcr2rlr * ((-X2rlr * np.sin(Arcr) + Y2rlr * np.cos(Arcr)) * 
                              (d2ArcrdAcmp2 - dLftdAcmp * dArcrdAcmp/R2rlr) +
                              (-X2rlr * np.cos(Arcr) - Y2rlr * np.sin(Arcr) + Rrcr2rlr) * 
                              dArcrdAcmp**2) / R2rlr
    d2u2dAcmp2 = Rrcr2rlr * ((np.cos(Arcr) + u2 * np.sin(Arcr)) * d2ArcrdAcmp2 +
                             (-np.sin(Arcr) + u2 * np.cos(Arcr)) * dArcrdAcmp**2 +
                             2 * np.sin(Arcr) * du2dAcmp * dArcrdAcmp) / X2rlr
    d2A2rlrdAcmp2 = d2u2dAcmp2/(1 + u2**2) - 2 * u2 * dA2rlrdAcmp**2
    d2AcmdAcmp2 = rot * d2A2rlrdAcmp2
    Acl = (np.pi/180)**2 * (d2LftdAcmp2 - Vel * 180/np.pi * d2AcmdAcmp2) / dAcmdAcmp**2
    
    # Calculate third derivatives
    d3ArcrdAcmp3 = -csvajo['j'].values * (180/np.pi)**2
    d3LftdAcmp3 = (Rrcr2rlr * ((-X2rlr * np.sin(Arcr) + Y2rlr * np.cos(Arcr)) * 
                               (d3ArcrdAcmp3 - dLftdAcmp * d2ArcrdAcmp2/R2rlr -
                                d2LftdAcmp2 * dArcrdAcmp/R2rlr + 
                                dLftdAcmp**2 * dArcrdAcmp/R2rlr**2 - dArcrdAcmp**3) +
                               (3 * d2ArcrdAcmp2 * dArcrdAcmp - 
                                dLftdAcmp * dArcrdAcmp**2/R2rlr) * 
                               (-X2rlr * np.cos(Arcr) - Y2rlr * np.sin(Arcr) + Rrcr2rlr)) -
                   dLftdAcmp * d2LftdAcmp2) / R2rlr
    d3u2dAcmp3 = Rrcr2rlr * ((np.cos(Arcr) + u2 * np.sin(Arcr)) * 
                             (d3ArcrdAcmp3 - dArcrdAcmp**3) +
                             3 * (-np.sin(Arcr) + u2 * np.cos(Arcr)) * 
                             dArcrdAcmp * d2ArcrdAcmp2 +
                             3 * np.sin(Arcr) * du2dAcmp * d2ArcrdAcmp2 +
                             3 * np.cos(Arcr) * du2dAcmp * dArcrdAcmp**2 +
                             3 * np.sin(Arcr) * d2u2dAcmp2 * dArcrdAcmp) / X2rlr
    d3A2rlrdAcmp3 = ((d3u2dAcmp3 - 2 * u2 * dA2rlrdAcmp * d2u2dAcmp2) / (1 + u2**2) -
                     2 * dA2rlrdAcmp**2 * du2dAcmp -
                     4 * u2 * dA2rlrdAcmp * d2A2rlrdAcmp2)
    d3AcmdAcmp3 = rot * d3A2rlrdAcmp3
    Jrk = (np.pi/180)**3 * ((d3LftdAcmp3 - d3AcmdAcmp3 * Vel * 180/np.pi) / 
                            dAcmdAcmp**3 - 
                            3 * d2AcmdAcmp2 * Acl * (180/np.pi)**2 / dAcmdAcmp**2)
    
    # Create output DataFrame
    return pd.DataFrame({
        'ca': Acmd,
        's': Lft,
        'v': Vel,
        'a': Acl,
        'j': Jrk
    })
