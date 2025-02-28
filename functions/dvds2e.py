import numpy as np

def dvds2e(cacbr: float, params: dict) -> tuple[float, dict]:
    """
    Calculate lift error for dv-d (dwell or constant velocity to dwell) segment.
    
    Args:
        cacbr: relative cam angle vs ca1 at which the accel is cut back (degcm)
        params: Dictionary containing:
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
            vr: velocity at start or end of segment (>=0, <lift units>/degcm)
            amx: max accel (<lift units>/degcm^2)
            dmx: max decel (>0, <lift units>/degcm^2)
            jmx: jerk (<lift units>/degcm^3)
    
    Returns:
        tuple: (s2e, params)
            s2e: lift error at end of segment (%)
            params: Updated dictionary with calculated parameters:
                dcaa: length of sub-segment A (degcm)
                dcab: length of sub-segment B (degcm)
                dcac: length of sub-segment C (degcm)
                dcad: length of sub-segment D (degcm)
                dcae: length of sub-segment E (degcm)
                sab: lift at end of segment A (<lift units>)
                sbc: lift at end of segment B (<lift units>)
                scd: lift at end of segment C (<lift units>)
                sde: lift at end of segment D (<lift units>)
                dsdcaab: velocity at AB transition (<lift units>/degcm)
                dsdcabc: velocity at BC transition (<lift units>/degcm)
                dsdcacd: velocity at CD transition (<lift units>/degcm)
                dsdcade: velocity at DE transition (<lift units>/degcm)
                d2sdca2bc: accel at BC transition (<lift units>/degcm^2)
                d2sdca2de: accel at DE transition (<lift units>/degcm^2)
    """
    # Calculate sub-segment A and B parameters
    if cacbr > params['amx'] / params['jmx']:
        params['dcaa'] = params['amx'] / params['jmx']
        params['dcab'] = cacbr - params['dcaa']
        params['d2sdca2bc'] = params['amx']
    else:
        params['dcaa'] = cacbr
        params['dcab'] = 0
        params['d2sdca2bc'] = params['jmx'] * cacbr
    
    # Calculate positions and velocities at transition points
    params['sab'] = params['jmx']/6 * params['dcaa']**3 + params['vr'] * params['dcaa'] + params['S1']
    params['dsdcaab'] = params['jmx']/2 * params['dcaa']**2 + params['vr']
    params['sbc'] = (params['amx']/2 * params['dcab']**2 + 
                     params['dsdcaab'] * params['dcab'] + params['sab'])
    params['dsdcabc'] = params['amx'] * params['dcab'] + params['dsdcaab']
    
    # Calculate sub-segment C parameters
    # d2sdca2CD=-DMX (may have segment D)
    dcac1 = (params['dmx'] + params['d2sdca2bc']) / params['jmx']
    # dsdca2=0 without segment D
    dcac2 = (params['d2sdca2bc'] / params['jmx'] + 
             np.sqrt(0.5 * (params['d2sdca2bc'] / params['jmx'])**2 + 
                    params['dsdcabc'] / params['jmx']))
    params['dcac'] = min(dcac1, dcac2)
    
    # Calculate positions and velocities for segment C
    params['scd'] = (-params['jmx']/6 * params['dcac']**3 + 
                     params['d2sdca2bc']/2 * params['dcac']**2 + 
                     params['dsdcabc'] * params['dcac'] + params['sbc'])
    params['dsdcacd'] = (-params['jmx']/2 * params['dcac']**2 + 
                         params['d2sdca2bc'] * params['dcac'] + params['dsdcabc'])
    
    # Calculate sub-segment D and E parameters
    if dcac1 < dcac2:
        params['dcad'] = params['dsdcacd'] / params['dmx'] - params['dmx'] / (2 * params['jmx'])
        params['d2sdca2de'] = -params['dmx']
    else:
        params['dcad'] = 0
        params['d2sdca2de'] = -params['jmx'] * params['dcac'] + params['d2sdca2bc']
    
    # Calculate final positions and velocities
    params['sde'] = (-params['dmx']/2 * params['dcad']**2 + 
                     params['dsdcacd'] * params['dcad'] + params['scd'])
    params['dsdcade'] = -params['dmx'] * params['dcad'] + params['dsdcacd']
    params['dcae'] = -params['d2sdca2de'] / params['jmx']
    
    # Calculate lift error
    s2e = ((params['jmx']/6 * params['dcae']**3 + 
            params['d2sdca2de']/2 * params['dcae']**2 + 
            params['dsdcade'] * params['dcae'] + params['sde']) / params['S2'] - 1)
    
    return s2e, params 