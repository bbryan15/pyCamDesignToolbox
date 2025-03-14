import numpy as np
import pandas as pd

def rf2cs(csvajr: pd.DataFrame, Rbc: float, Rrlr: float, rot: int) -> pd.DataFrame:
    """
    Converts from reciprocating follower to cam surface.
    
    Args:
        csvajr: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: reciprocating follower cam angle vector (degcm)
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2)
            j: reciprocating follower jerk vector (mm/degcm^3) NOT USED
        Rbc: cam base circle radius (mm)
        Rrlr: roller follower radius (mm)
        rot: +1 clockwise, -1 counter-clockwise
    
    Returns:
        DataFrame with columns ['Atand', 'Rtan', 'dRtan']
            Atand: cam profile angle vector (deg)
            Rtan: cam radius at the point of tangent contact (mm)
            dRtan: dRtan/dAtand (mm/deg)
    """
    # Extract input vectors
    Acmd = csvajr['ca'].values
    Lft = csvajr['s'].values
    Vel = csvajr['v'].values
    Acl = csvajr['a'].values
    
    # Calculate distances and angles
    R2rlr = Lft + Rbc + Rrlr  # distance from cam axis to roller axis (mm)
    Aprsr = rot * np.arctan(Vel * 180/np.pi / R2rlr)  # reciprocating follower pressure angle (rad)
    Rtan = np.sqrt(R2rlr**2 + Rrlr**2 - 2*R2rlr*Rrlr*np.cos(Aprsr))  # cam radius at tangent contact
    
    # Calculate angle from cam profile reference
    cos_term = np.minimum(1, (R2rlr**2 + Rtan**2 - Rrlr**2)/(2*R2rlr*Rtan))
    Atan = Acmd*np.pi/180 + np.sign(Vel)*np.arccos(cos_term)
    Atan = rot * Atan  # sign change to properly render cam surface
    Atand = Atan * 180/np.pi
    
    # Calculate exact derivative
    dRtand = np.zeros_like(Acmd)
    for i in range(len(Acmd)):
        u2 = R2rlr[i]**2 + Rrlr**2 - 2*R2rlr[i]*Rrlr*np.cos(Aprsr[i])
        u3 = Vel[i] * 180/np.pi / R2rlr[i]
        dAprsrdAcmd = 180/np.pi * rot/(1 + u3**2) * (Acl[i]/R2rlr[i] - (Vel[i]/R2rlr[i])**2)
        dRtandAcmd = u2**(-0.5) * (R2rlr[i]*Vel[i] + 
                                  Rrlr*(R2rlr[i]*np.sin(Aprsr[i])*dAprsrdAcmd - 
                                       np.cos(Aprsr[i])*Vel[i]))
        
        u1 = (R2rlr[i]**2 + Rtan[i]**2 - Rrlr**2)/(2*R2rlr[i]*Rtan[i])
        if u1 >= 1:
            dAtanddAcmd = rot
        else:
            dAtanddAcmd = rot * (1 - 180/np.pi * np.sign(Vel[i]) * 
                                (1 - u1**2)**(-0.5) * 
                                (Vel[i]*(1/Rtan[i] - u1/R2rlr[i]) + 
                                 dRtandAcmd*(1/R2rlr[i] - u1/Rtan[i])))
        
        dRtand[i] = dRtandAcmd/dAtanddAcmd  # dRtan/dAtand (mm/deg)
    
    # Create output DataFrame
    return pd.DataFrame({
        'Atand': Atand,
        'Rtan': Rtan,
        'dRtan': dRtand
    })