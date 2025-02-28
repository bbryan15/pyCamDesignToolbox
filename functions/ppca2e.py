from scipy.optimize import root_scalar
from .pps2e import pps2e

def ppca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for p-p (peak accel to peak decel) segment.
    
    Args:
        J: jerk (<lift units>/degcm^3)
        params: Dictionary containing:
            ca1: Initial cam angle (degcm)
            ca2t: Target cam angle at end of segment (degcm)
            dca: Cam angle step (degcm)
            dcaa: Length of sub-segment A (degcm)
            dcab: Length of sub-segment B (degcm)
            dcac: Length of sub-segment C (degcm)
    
    Returns:
        tuple: (ca2e, params)
            ca2e: cam angle error at end of segment (degcm)
            params: Updated dictionary with calculated parameters
    """
    # Store J for use in pps2e
    params['jmx'] = J
    
    # Initialize dcaa
    dcaa = 0
    
    # Get initial s2err
    s2err, params = pps2e(dcaa, params)
    
    if s2err > 0:
        dcaamx = dcaa
        dcaamn = dcaa
        while s2err >= 0:
            dcaamn = dcaamn - params['dca']
            s2err, params = pps2e(dcaamn, params)
        
        # Find zero crossing using scipy's root_scalar (equivalent to MATLAB's fzero)
        sol = root_scalar(lambda x: pps2e(x, params)[0],
                         bracket=[dcaamn, dcaamx],
                         method='brentq')
        dcaa = sol.root
        
    elif s2err < 0:
        dcaamn = dcaa
        dcaamx = dcaa
        while s2err <= 0:
            dcaamx = dcaamx + params['dca']
            s2err, params = pps2e(dcaamx, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: pps2e(x, params)[0],
                         bracket=[dcaamn, dcaamx],
                         method='brentq')
        dcaa = sol.root
    
    # Calculate final cam angle error
    ca2e = params['ca1'] + params['dcaa'] + params['dcab'] + params['dcac'] - params['ca2t']
    
    return ca2e, params 