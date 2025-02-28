import numpy as np
from .ppca2e import ppca2e
from .pps2e import pps2e

def gcam0_pp(params: dict) -> np.ndarray:
    """
    Generate cam profile for p-p (peak accel to peak decel) segment.
    
    Args:
        params: Dictionary containing:
            ca1: Initial cam angle (degcm)
            dca: Cam angle step (degcm)
            S1: Initial lift (<lift units>)
            S2: Final lift (<lift units>)
            amx: Max accel limit (<lift units>/degcm^2)
            dmx: Max decel limit (>0, <lift units>/degcm^2)
            jmx: Max jerk (<lift units>/degcm^3)
    
    Returns:
        np.ndarray: csvaj matrix with columns [ca, s, v, a, j]
    """
    # Store original parameters
    J = params['jmx']
    
    # Initialize dcaa search
    dcaa = 0
    s2err, params = pps2e(dcaa, params.copy())
    
    # If s2err is negative, search forward
    if s2err < 0:
        dcaa_min = dcaa
        dcaa_step = params['dca']
        while True:
            dcaa += dcaa_step
            s2err, _ = pps2e(dcaa, params.copy())
            if s2err > 0:
                dcaa_max = dcaa
                break
            if dcaa > 100:  # Safety limit
                raise ValueError("Failed to find positive s2err")
    # If s2err is positive, search backward
    else:
        dcaa_max = dcaa
        dcaa_step = -params['dca']
        while True:
            dcaa += dcaa_step
            s2err, _ = pps2e(dcaa, params.copy())
            if s2err < 0:
                dcaa_min = dcaa
                break
            if dcaa < -100:  # Safety limit
                raise ValueError("Failed to find negative s2err")
    
    # Binary search for zero crossing
    while abs(dcaa_max - dcaa_min) > 1e-10:
        dcaa = (dcaa_min + dcaa_max) / 2
        s2err, params = pps2e(dcaa, params.copy())
        if s2err > 0:
            dcaa_max = dcaa
        else:
            dcaa_min = dcaa
    
    # Calculate final parameters with best dcaa
    _, params = pps2e(dcaa, params)
    
    # Generate csvaj matrix
    ca2 = params['ca1'] + params['dcaa'] + params['dcab'] + params['dcac']
    n = round((ca2 - params['ca1']) / params['dca']) + 1
    csvaj = np.zeros((n, 5))
    
    for i in range(n):
        ca = params['ca1'] + params['dca'] * i
        csvaj[i, 0] = ca
        
        # Sub-segment A (optional): constant acceleration
        if ca <= params['ca1'] + params['dcaa'] and params['dcaa'] > 0:
            carel = ca - params['ca1']
            csvaj[i, 1] = params['amx']/2 * carel**2 + params['S1']
            csvaj[i, 2] = params['amx'] * carel
            csvaj[i, 3] = params['amx']
            csvaj[i, 4] = 0
            
        # Sub-segment B: decreasing acceleration
        elif ca <= params['ca1'] + params['dcaa'] + params['dcab']:
            carel = ca - (params['ca1'] + params['dcaa'])
            csvaj[i, 1] = (-params['jmx']/6 * carel**3 + 
                          params['d2sdca2ab']/2 * carel**2 +
                          params['dsdcaab'] * carel + params['sab'])
            csvaj[i, 2] = (-params['jmx']/2 * carel**2 +
                          params['d2sdca2ab'] * carel + params['dsdcaab'])
            csvaj[i, 3] = -params['jmx'] * carel + params['d2sdca2ab']
            csvaj[i, 4] = -params['jmx']
            
        # Sub-segment C (optional): constant deceleration
        else:
            carel = ca - (params['ca1'] + params['dcaa'] + params['dcab'])
            csvaj[i, 1] = (-params['dmx']/2 * carel**2 +
                          params['dsdcabc'] * carel + params['sbc'])
            csvaj[i, 2] = -params['dmx'] * carel + params['dsdcabc']
            csvaj[i, 3] = -params['dmx']
            csvaj[i, 4] = 0
    
    return csvaj 