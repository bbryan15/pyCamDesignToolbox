import numpy as np
from typing import Tuple

def dvsvas2e(cacbr: float, params: dict) -> Tuple[float, dict]:
    """
    Calculate lift error for dv-sva (dwell or constant velocity to specified velocity and acceleration) segment.
    
    Args:
        cacbr: relative cam angle vs ca1 at which the accel is cut back (degcm)
        params: Dictionary containing:
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
            vr: velocity at start or end of segment (>=0, <lift units>/degcm)
            vmatch: final velocity (>=0, <lift units>/degcm)
            amatch: final accel (=>0, <lift units>/degcm^2)
            amx: max accel (<lift units>/degcm^2)
            dmx: max decel (>0, <lift units>/degcm^2)
            jmx: jerk (<lift units>/degcm^3)
    
    Returns:
        tuple: (s2e, params)
            s2e: lift error at end of segment (%)
            params: Updated dictionary with calculated parameters
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
    
    # Choose dcad to match final velocity to vmatch
    # dcad = length of segment D or negative for partial segment C with d2sdca2de > -dmx (degcm)
    a = params['jmx']
    b = 2 * params['dmx']
    c = (-params['vr'] - params['d2sdca2bc'] * params['dcab'] - 
         (params['d2sdca2bc']**2 + 0.5 * params['amatch']**2 - params['dmx']**2) / params['jmx'] + 
         params['vmatch'])
    
    dcad = -c / params['dmx']  # dcac > 0 result
    if dcad > 0:
        params['dcad'] = dcad
        params['d2sdca2de'] = -params['dmx']
    else:
        arg = b**2 - 4*a*c
        if arg < 0:
            print("Warning: gcam0 function, 'dv-sva' option: Unable to match v2 for current dcaa iteration. Continuing with d2sdca2de=0.")
        
        # dcad < 0 result: physical root for continuity with dcac>0 result
        # b^2-4*a*c=0 corresponds to min dcad=-dmx/jmx for d2sdca2de = 0
        dcad = (-b + np.sqrt(max(arg, 0))) / (2*a)
        params['dcad'] = 0
        params['d2sdca2de'] = -params['dmx'] - dcad * params['jmx']
    
    # Calculate remaining parameters
    params['dcac'] = (params['d2sdca2bc'] - params['d2sdca2de']) / params['jmx']
    params['scd'] = (-params['jmx']/6 * params['dcac']**3 + 
                     params['d2sdca2bc']/2 * params['dcac']**2 + 
                     params['dsdcabc'] * params['dcac'] + params['sbc'])
    params['dsdcacd'] = (-params['jmx']/2 * params['dcac']**2 + 
                         params['d2sdca2bc'] * params['dcac'] + params['dsdcabc'])
    params['sde'] = (-params['dmx']/2 * params['dcad']**2 + 
                     params['dsdcacd'] * params['dcad'] + params['scd'])
    params['dsdcade'] = -params['dmx'] * params['dcad'] + params['dsdcacd']
    params['dcae'] = (params['amatch'] - params['d2sdca2de']) / params['jmx']
    
    # Calculate lift error
    s2e = ((params['jmx']/6 * params['dcae']**3 + 
            params['d2sdca2de']/2 * params['dcae']**2 + 
            params['dsdcade'] * params['dcae'] + params['sde']) / params['S2'] - 1)
    
    return s2e, params 