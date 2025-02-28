def dvca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for d-v (dwell to constant velocity) segment.
    
    Args:
        J: max jerk (<lift units>/degcm^3)
        params: Dictionary containing:
            ca1: cam angle at start of segment (degcm)
            ca2t: target cam angle at end of segment (degcm)
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
            vr: velocity at start or end of segment (>=0, <lift units>/degcm)
            amx: max accel (<lift units>/degcm^2)
    
    Returns:
        tuple: (ca2e, params)
            ca2e: cam angle error at end of segment (degcm)
            params: Updated dictionary with calculated parameters
    """
    # Calculate sub-segment lengths and parameters
    if params['vr'] > params['amx']**2 / J:
        params['dcaa'] = params['amx'] / J
        params['dcab'] = params['vr'] / params['amx'] - params['amx'] / J
    else:
        params['dcaa'] = (params['vr'] / J)**0.5
        params['dcab'] = 0
    
    # Calculate positions, velocities, and accelerations at transition points
    params['sab'] = J/6 * params['dcaa']**3 + params['S1']
    params['dsdcaab'] = J/2 * params['dcaa']**2
    
    params['sbc'] = (params['amx']/2 * params['dcab']**2 + 
                     params['dsdcaab'] * params['dcab'] + params['sab'])
    params['dsdcabc'] = params['amx'] * params['dcab'] + params['dsdcaab']
    params['d2sdca2bc'] = J * params['dcaa']
    
    params['dcac'] = params['d2sdca2bc'] / J
    params['scd'] = (-J/6 * params['dcac']**3 + 
                     params['d2sdca2bc']/2 * params['dcac']**2 + 
                     params['dsdcabc'] * params['dcac'] + params['sbc'])
    
    params['dcad'] = (params['S2'] - params['scd']) / params['vr']
    
    # Calculate cam angle error
    ca2e = (params['ca1'] + params['dcaa'] + params['dcab'] + 
            params['dcac'] + params['dcad'] - params['ca2t'])
    
    return ca2e, params 