import numpy as np
import pandas as pd

def rf2of(csvajr: pd.DataFrame, Rbc: float, Rrcr2rlr: float, Rrlr: float, 
          Xrcr: float, Yrcr: float, rot: int) -> tuple[pd.DataFrame, float]:
    """
    Converts from reciprocating follower to oscillating follower.
    
    Args:
        csvajr: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: reciprocating follower cam angle vector (degcm)
            s: reciprocating follower lift vector (mm)
            v: reciprocating follower velocity vector (mm/degcm)
            a: reciprocating follower acceleration vector (mm/degcm^2)
            j: reciprocating follower jerk vector (mm/degcm^3)
        Rbc: cam base circle radius (mm)
        Rrcr2rlr: distance from rocker axis to roller axis (mm)
        Rrlr: roller follower radius (mm)
        Xrcr: horizontal distance from cam axis to rocker axis (mm)
        Yrcr: vertical distance from cam axis to rocker axis (mm)
        rot: +1 clockwise, -1 counter-clockwise, looking at the cam with the 
             valve on the right or initial rocker lift clockwise
    
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
    # Convert cam angle to radians and calculate distances
    Acm = csvajr['ca'].values * np.pi/180
    R2rlr = csvajr['s'].values + Rbc + Rrlr
    R2rcr = np.sqrt(Xrcr**2 + Yrcr**2)
    A2rcr = np.arctan2(Yrcr, Xrcr)
    
    # Calculate angles
    cos_term = np.minimum(1, (R2rlr**2 + R2rcr**2 - Rrcr2rlr**2)/(2*R2rlr*R2rcr))
    A2rlr = A2rcr + np.arccos(cos_term)
    Acmp = Acm + rot*(A2rlr[0] - A2rlr)
    
    # Calculate rocker angle
    Arcr = np.arctan2(R2rlr*np.sin(A2rlr) - Yrcr, 
                      R2rlr*np.cos(A2rlr) - Xrcr)
    Arcr = np.mod(Arcr, 2*np.pi)
    dArcr = Arcr[0] - Arcr
    
    # Convert angles to degrees
    Acmpd = Acmp * 180/np.pi
    dArcrd = dArcr * 180/np.pi
    
    # Calculate derivatives
    u1n = R2rlr*np.sin(A2rlr) - Yrcr
    u1d = R2rlr*np.cos(A2rlr) - Xrcr
    u1 = u1n/u1d
    u2 = (R2rlr**2 + R2rcr**2 - Rrcr2rlr**2)/(2*R2rlr*R2rcr)
    
    # First derivatives
    du2dLft = 1/R2rcr - u2/R2rlr
    dA2rlrdLft = -(1 - u2**2)**(-0.5) * du2dLft
    du1ndLft = R2rlr*np.cos(A2rlr)*dA2rlrdLft + np.sin(A2rlr)
    du1ddLft = -R2rlr*np.sin(A2rlr)*dA2rlrdLft + np.cos(A2rlr)
    du1dLft = (u1d*du1ndLft - u1n*du1ddLft)/u1d**2
    
    dArcrdAcm = du1dLft/(1 + u1**2) * csvajr['v'].values * 180/np.pi
    dAcmpdAcm = 1 - rot*dA2rlrdLft * csvajr['v'].values * 180/np.pi
    Avel = -dArcrdAcm/dAcmpdAcm
    
    # Second derivatives
    d2A2rlrdLft2 = ((1 - u2**2)**(-0.5) * (du2dLft/R2rlr - u2/R2rlr**2) - 
                    (1 - u2**2)**(-1.5) * u2 * du2dLft**2)
    d2AcmpdAcm2 = -rot*(dA2rlrdLft * csvajr['a'].values * (180/np.pi)**2 + 
                        d2A2rlrdLft2 * (csvajr['v'].values * 180/np.pi)**2)
    
    d2u1ndLft2 = (R2rlr*np.cos(A2rlr)*d2A2rlrdLft2 + 
                  (du1ddLft + np.cos(A2rlr))*dA2rlrdLft)
    d2u1ddLft2 = (-R2rlr*np.sin(A2rlr)*d2A2rlrdLft2 - 
                  (du1ndLft + np.sin(A2rlr))*dA2rlrdLft)
    d2u1dLft2 = ((u1d*d2u1ndLft2 - u1n*d2u1ddLft2)/u1d**2 - 
                 2*du1dLft*du1ddLft/u1d)
    
    d2ArcrdAcm2 = ((du1dLft*csvajr['a'].values*(180/np.pi)**2 + 
                    d2u1dLft2*(csvajr['v'].values*180/np.pi)**2)/(1 + u1**2) - 
                   2*u1*(du1dLft*csvajr['v'].values*180/np.pi/(1 + u1**2))**2)
    
    Aacl = -np.pi/180*(d2ArcrdAcm2/dAcmpdAcm**2 - 
                       dArcrdAcm*d2AcmpdAcm2/dAcmpdAcm**3)
    
    # Third derivatives (continued in next part due to length) 