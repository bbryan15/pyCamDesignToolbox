import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def derv1(ca: np.ndarray, x: np.ndarray, caw: float) -> np.ndarray:
    """
    Computes first-order derivative by central difference approximation.
    
    Uses the formula: fp = (f(x+h)-f(x-h))/(2*h) + O(h)
    
    Args:
        ca: cam profile angle (assumed equally spaced)
        x: cam profile variable vector
        caw: wrapping delta angle (360 or 720, governs treatment of end points)
    
    Returns:
        numpy.ndarray: dx/dca vector
    """
    nr = len(ca)
    h = (ca[-1] - ca[0]) / (nr - 1)
    xp = np.zeros(nr)
    
    # Calculate central differences for interior points
    for i in range(1, nr-1):
        xp[i] = (x[i+1] - x[i-1]) / (2*h)
    
    # Handle endpoints based on wrapping condition
    if ca[-1] == ca[0] + caw:  # full cam; duplicate point at end
        xp[0] = (x[1] - x[-2]) / (2*h)
        xp[-1] = xp[0]
    elif ca[-1] == ca[0] + caw - h:  # full cam; no duplicate point at end
        xp[0] = (x[1] - x[-1]) / (2*h)
        xp[-1] = (x[0] - x[-2]) / (2*h)
    else:  # partial cam; assume dwells at ends equal to endpoints
        xp[0] = (x[1] - x[0]) / (2*h)
        xp[-1] = (x[-1] - x[-2]) / (2*h)
    
    return xp

def derv2(ca: np.ndarray, x: np.ndarray, caw: float) -> np.ndarray:
    """
    Computes second-order derivative by central difference approximation.
    
    Uses the formula: fpp = (f(x+h)-2*f(x)+f(x-h))/h^2 + O(h)
    
    Args:
        ca: cam profile angle (assumed equally spaced)
        x: cam profile variable vector
        caw: wrapping delta angle (360 or 720, governs treatment of end points)
    
    Returns:
        numpy.ndarray: d2x/dca2 vector
    """
    nr = len(ca)
    h = (ca[-1] - ca[0]) / (nr - 1)
    x2p = np.zeros(nr)
    
    # Calculate central differences for interior points
    for i in range(1, nr-1):
        x2p[i] = (x[i+1] - 2*x[i] + x[i-1]) / h**2
    
    # Handle endpoints based on wrapping condition
    if ca[-1] == ca[0] + caw:  # full cam; duplicate point at end
        x2p[0] = (x[1] - 2*x[0] + x[-2]) / h**2
        x2p[-1] = x2p[0]
    elif ca[-1] == ca[0] + caw - h:  # full cam; no duplicate point at end
        x2p[0] = (x[1] - 2*x[0] + x[-1]) / h**2
        x2p[-1] = (x[0] - 2*x[-1] + x[-2]) / h**2
    else:  # partial cam; assume dwells at ends equal to endpoints
        x2p[0] = (x[1] - 2*x[0] + x[0]) / h**2
        x2p[-1] = (x[-1] - 2*x[-1] + x[-2]) / h**2
    
    return x2p

def derv3(ca: np.ndarray, x: np.ndarray, caw: float) -> np.ndarray:
    """
    Computes third-order derivative by central difference approximation.
    
    Uses the formula: fppp = (f(x+2*h)-2*f(x+h)+2*f(x-h)-f(x-2*h))/(2*h^3) + O(h)
    
    Args:
        ca: cam profile angle (assumed equally spaced)
        x: cam profile variable vector
        caw: wrapping delta angle (360 or 720, governs treatment of end points)
    
    Returns:
        numpy.ndarray: d3x/dca3 vector
    """
    nr = len(ca)
    h = (ca[-1] - ca[0]) / (nr - 1)
    x3p = np.zeros(nr)
    
    # Calculate central differences for interior points
    for i in range(2, nr-2):
        x3p[i] = (x[i+2] - 2*x[i+1] + 2*x[i-1] - x[i-2]) / (2*h**3)
    
    # Handle endpoints based on wrapping condition
    if ca[-1] == ca[0] + caw:  # full cam; duplicate point at end
        x3p[0] = (x[2] - 2*x[1] + 2*x[-2] - x[-3]) / (2*h**3)
        x3p[1] = (x[3] - 2*x[2] + 2*x[0] - x[-2]) / (2*h**3)
        x3p[-2] = (x[1] - 2*x[0] + 2*x[-3] - x[-4]) / (2*h**3)
        x3p[-1] = x3p[0]
    elif ca[-1] == ca[0] + caw - h:  # full cam; no duplicate point at end
        x3p[0] = (x[2] - 2*x[1] + 2*x[-1] - x[-2]) / (2*h**3)
        x3p[1] = (x[3] - 2*x[2] + 2*x[0] - x[-1]) / (2*h**3)
        x3p[-2] = (x[0] - 2*x[-1] + 2*x[-3] - x[-4]) / (2*h**3)
        x3p[-1] = (x[1] - 2*x[0] + 2*x[-2] - x[-3]) / (2*h**3)
    else:  # partial cam; assume dwells at ends equal to endpoints
        x3p[0] = (x[2] - 2*x[1] + 2*x[0] - x[0]) / (2*h**3)
        x3p[1] = (x[3] - 2*x[2] + 2*x[0] - x[0]) / (2*h**3)
        x3p[-2] = (x[-1] - 2*x[-1] + 2*x[-3] - x[-4]) / (2*h**3)
        x3p[-1] = (x[-1] - 2*x[-1] + 2*x[-2] - x[-3]) / (2*h**3)
    
    return x3p

def allderv(cs: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms cam angle, lift DataFrame to cam angle, lift, velocity, accel, jerk DataFrame.
    
    Args:
        cs: DataFrame with columns ['ca', 's']
            ca: cam angle vector (degcm)
            s: lift vector (mm or deg)
    
    Returns:
        pd.DataFrame: DataFrame with columns ['ca', 's', 'v', 'a', 'j']
            ca: cam angle vector (degcm)
            s: lift vector (mm or deg)
            v: velocity vector (mm/degcm or deg/degcm)
            a: accel vector (mm/degcm^2 or deg/degcm^2)
            j: jerk vector (mm/degcm^3 or deg/degcm^3)
    """
    # Extract vectors
    Acmd = cs['ca'].values  # cam rotation angle (deg)
    Lft = cs['s'].values    # lift at original breakpoints (mm or deg)
    
    # Create equally spaced points
    npts = len(Acmd)
    dAcmd = (Acmd[-1] - Acmd[0]) / (npts - 1)  # equally-spaced cam angle increment
    Acmde = np.linspace(Acmd[0], Acmd[-1], npts)  # equally-spaced cam angle
    
    # Interpolate lift to equally spaced points
    interp_func = interp1d(Acmd, Lft, kind='cubic', bounds_error=False)
    Lfte = interp_func(Acmde)
    
    # Calculate derivatives at equally spaced points
    Vele = derv1(Acmde, Lfte, 360)  # velocity
    Acle = derv2(Acmde, Lfte, 360)  # acceleration
    Jrke = derv3(Acmde, Lfte, 360)  # jerk
    
    # Create output DataFrame
    return pd.DataFrame({
        'ca': Acmd,
        's': Lft,
        'v': interp1d(Acmde, Vele, kind='cubic', bounds_error=False)(Acmd),
        'a': interp1d(Acmde, Acle, kind='cubic', bounds_error=False)(Acmd),
        'j': interp1d(Acmde, Jrke, kind='cubic', bounds_error=False)(Acmd)
    }) 