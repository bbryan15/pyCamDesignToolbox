from scipy.optimize import root_scalar
from .dvps2e import dvps2e

def dvpca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for dv-p segment.
    
    Args:
        J: max jerk (lift units/degcm^3)
        params: dictionary containing:
            ca1: cam angle at start of segment (degcm)
            ca2t: target cam angle at end of segment (degcm)
            dca: cam angle step (degcm)
            amx: max acceleration
            dcaa, dcab, dcac, dcad: segment lengths
    
    Returns:
        tuple: (ca2e, updated_params)
            ca2e: cam angle error at end of segment
            updated_params: dictionary with updated values
    """
    # Store jerk value
    params['jmx'] = J
    
    # Calculate ratio of max accel to jerk
    cacbr = params['amx'] / J
    
    # Calculate lift error
    s2err, params = dvps2e(cacbr, params)
    
    if s2err > 0:
        # Error is positive, search downward
        cacbrmx = cacbr
        cacbrmn = cacbr
        while s2err >= 0:
            cacbrmn = cacbrmn - params['dca']
            s2err, params = dvps2e(cacbrmn, params)
        # Find zero crossing using root_scalar
        sol = root_scalar(lambda x: dvps2e(x, params)[0],  # Only use s2err for root finding
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        _, params = dvps2e(cacbr, params)  # Get final params after finding root
        
    elif s2err < 0:
        # Error is negative, search upward
        cacbrmn = cacbr
        cacbrmx = cacbr
        while s2err <= 0:
            cacbrmx = cacbrmx + params['dca']
            s2err, params = dvps2e(cacbrmx, params)
        # Find zero crossing
        sol = root_scalar(lambda x: dvps2e(x, params)[0],  # Only use s2err for root finding
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        _, params = dvps2e(cacbr, params)  # Get final params after finding root
    
    # Calculate final cam angle error
    ca2e = (params['ca1'] + params['dcaa'] + params['dcab'] + 
            params['dcac'] + params['dcad'] - params['ca2t'])
    
    return ca2e, params 