import numpy as np
import pandas as pd

def convel(ca1: float, ca2: float, dca: float, s1: float, s2: float, vc: float, depvar: str) -> pd.DataFrame:
    # Input validation
    if ca1 >= ca2:
        raise ValueError("ca1 must be less than ca2")
    
    if dca <= 0:
        raise ValueError("dca must be greater than 0")
    
    # Check if (ca2-ca1) is divisible by dca
    if not abs((ca2 - ca1) / dca - round((ca2 - ca1) / dca)) < 1e-10:
        raise ValueError("(ca2-ca1) must be divisible by dca")
        
    if depvar not in ['s2', 'vc']:
        raise ValueError("depvar must be either 's2' or 'vc'")
    
    # Generate cam angle series
    ca_series = np.arange(ca1, ca2 + dca, dca)
    
    # Calculate constant velocity based on depvar
    if depvar == 'vc':
        vc = (s2 - s1) / (ca2 - ca1)  

    df = pd.DataFrame({
        "ca": ca_series,
        "v": vc,   
        "a": 0,    
        "j": 0     
    })
    
    # Calculate displacement (s) using cumulative sum
    df["s"] = s1 + (df["ca"] - ca1) * df["v"]
    
    
    # Reorder columns to match required format
    return df[["ca", "s", "v", "a", "j"]] 
    