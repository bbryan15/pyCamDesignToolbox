import pandas as pd

def witol(a: float, b: float, rtol: float, atol: float) -> bool:
    """
    Check if two scalars are equal within a specified tolerance.
    
    Args:
        a: First scalar to compare
        b: Second scalar to compare
        rtol: Relative (fractional) tolerance
        atol: Absolute tolerance
    
    Returns:
        bool: True if within tolerance, False if not
    """
    # First check absolute tolerance
    if abs(a - b) > atol:
        return False
    
    # Then check relative tolerance if values are large enough
    num = max(abs(a), abs(b))
    if num > atol:
        if abs(a - b)/num > rtol:
            return False
    
    return True

def concat(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates multiple cam profile segments.
    
    Args:
        segments: List of pandas DataFrames where each DataFrame has columns:
            ca: cam angle (degcm)
            s: lift (mm for reciprocating follower, degr for oscillating rocker follower)
            v: velocity (mm/degcm or degr/degcm)
            a: acceleration (mm/degcm^2 or degr/degcm^2)
            j: jerk (mm/degcm^3 or degr/degcm^3)
    
    Returns:
        pd.DataFrame: Concatenated cam profile with columns ['ca', 's', 'v', 'a', 'j']
    
    Raises:
        ValueError: If segments have mismatched values at connection points
    """
    if not segments:
        return pd.DataFrame(columns=['ca', 's', 'v', 'a', 'j'])
    
    # Start with first segment
    result = segments[0].copy()
    
    # Concatenate remaining segments
    for i, seg in enumerate(segments[1:], 1):
        # Get last row of current result and first row of new segment
        last = result.iloc[-1]
        first = seg.iloc[0]
        
        # Check for mismatches
        if last['ca'] != first['ca']:
            raise ValueError(f'Cam angle mismatch for segments {i-1} and {i}')
        
        if not witol(last['s'], first['s'], 1e-4, 2e-6):
            raise ValueError(f'Cam lift mismatch for segments {i-1} and {i}')
            
        if not witol(last['v'], first['v'], 1e-4, 1e-6):
            raise ValueError(f'Cam velocity mismatch for segments {i-1} and {i}')
            
        if not witol(last['a'], first['a'], 1e-4, 5e-7):
            raise ValueError(f'Cam acceleration mismatch for segments {i-1} and {i}')
        
        # Append new segment (excluding first row)
        result = pd.concat([result, seg.iloc[1:]], ignore_index=True)
        
        # Check and potentially adjust jerk at connection point
        if not witol(last['j'], first['j'], 1e-4, 1e-8):
            nc = len(result) - len(seg.iloc[1:]) - 1  # Index of connection point
            result.loc[nc, 'j'] = ((result.loc[nc+1, 'a'] - result.loc[nc-1, 'a']) / 
                                 (result.loc[nc+1, 'ca'] - result.loc[nc-1, 'ca']))
    
    return result 