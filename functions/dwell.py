import numpy as np
import pandas as pd

# create a function to generate a dwell profile, starting at ca1, ending at ca2, with a step of dca, and a constant lift of sc
def dwell(ca1: float, ca2: float, dca: float, sc: float) -> pd.DataFrame:
    # Input validation
    if ca1 >= ca2:
        raise ValueError("ca1 must be less than ca2")
    
    if dca <= 0:
        raise ValueError("dca must be greater than 0")
    
    # Check if (ca2-ca1) is divisible by dca
    if not abs((ca2 - ca1) / dca - round((ca2 - ca1) / dca)) < 1e-10:
        raise ValueError("(ca2-ca1) must be divisible by dca")
    
    # Generate the cam angle series from ca1 to ca2 with dca step
    ca_series = pd.Series(np.arange(ca1, ca2 + dca, dca))
    
    return pd.DataFrame({
        "ca": ca_series,
        "s": sc,  
        "v": 0,   
        "a": 0,   
        "j": 0    
    })


