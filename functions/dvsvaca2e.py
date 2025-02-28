from scipy.optimize import root_scalar
from .dvsvas2e import dvsvas2e

def dvsvaca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for dv-sva (dwell or constant velocity to specified velocity and acceleration) segment.
    
    Args:
        J: max jerk (lift units/degcm^3)
        params: Dictionary containing:
            ca1: cam angle at start of segment (degcm)
            ca2t: target cam angle at end of segment (degcm)
            dca: cam angle step (degcm): typically 0.5 or 1.0
            dcaa: length of sub-segment A (degcm)
            dcab: length of sub-segment B (degcm)
            dcac: length of sub-segment C (degcm)
            dcad: length of sub-segment D (degcm)
            dcae: length of sub-segment E (degcm)
            amx: max accel limit (<lift units>/degcm^2)
    
    Returns:
        tuple: (ca2e, params)
            ca2e: cam angle error at end of segment (degcm)
            params: Updated dictionary with calculated parameters
    """
    # Store J for use in dvsvas2e
    params['jmx'] = J
    
    # Calculate initial cacbr (cam angle change before reversal)
    cacbr = params['amx'] / J
    
    # Get initial s2err
    s2err, params = dvsvas2e(cacbr, params)
    
    if s2err > 0:
        cacbrmx = cacbr
        cacbrmn = cacbr
        while s2err >= 0:
            cacbrmn = cacbrmn - params['dca']
            s2err, params = dvsvas2e(cacbrmn, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: dvsvas2e(x, params)[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        
    elif s2err < 0:
        cacbrmn = cacbr
        cacbrmx = cacbr
        while s2err <= 0:
            cacbrmx = cacbrmx + params['dca']
            s2err, params = dvsvas2e(cacbrmx, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: dvsvas2e(x, params)[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
    
    # Calculate final cam angle error
    ca2e = (params['ca1'] + params['dcaa'] + params['dcab'] + 
            params['dcac'] + params['dcad'] + params['dcae'] - params['ca2t'])
    
    return ca2e, params 