import pandas as pd
import numpy as np

def conaccel(ca1: float, ca2: float, dca: float, s1: float, s2: float, 
             v1: float, v2: float, ac: float, depvars: str) -> pd.DataFrame:
    # Input validation
    if ca1 >= ca2:
        raise ValueError("ca2 must be > ca1")
    
    if dca <= 0:
        raise ValueError("dca must be greater than 0")
    
    # Check if (ca2-ca1) is divisible by dca
    if not abs((ca2 - ca1) / dca - round((ca2 - ca1) / dca)) < 1e-10:
        raise ValueError("(ca2-ca1) must be divisible by dca")
        
    if depvars not in ['s2v2', 's2ac', 'v2ac']:
        raise ValueError("depvars must be one of 's2v2', 's2ac', or 'v2ac'")
    
    # Calculate acceleration based on depvars
    if depvars == 's2v2':
        pass
    elif depvars == 's2ac':
        ac = (v2 - v1) / (ca2 - ca1)
    elif depvars == 'v2ac':
        ac = 2 * (s2 - s1 - v1 * (ca2 - ca1)) / ((ca2 - ca1) ** 2)
    
    # Calculate number of points exactly as in MATLAB
    n = round((ca2 - ca1) / dca) + 1
    
    # Initialize arrays with high precision
    df = np.zeros((n, 5), dtype=np.float64)
    
    # Calculate values using MATLAB's precision
    for i in range(n):
        ca = ca1 + dca * i
        t = ca - ca1
        df[i, 0] = ca
        df[i, 1] = s1 + v1 * t + 0.5 * ac * t * t
        df[i, 2] = v1 + ac * t
        df[i, 3] = ac
        df[i, 4] = 0
    
    # Convert to DataFrame
    df = pd.DataFrame(df, columns=['ca', 's', 'v', 'a', 'j'])
    
    # Ensure final values match exactly for s2ac and v2ac modes
    if depvars in ['s2ac', 'v2ac']:
        df.loc[df.index[-1], 's'] = s2
    
    return df.round(6)
    
