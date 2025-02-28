import numpy as np

def pps2e(dcaa: float, params: dict) -> tuple[float, dict]:
    """
    Calculate lift error for p-p (peak accel to peak decel) segment.
    Matches MATLAB implementation exactly.
    
    Args:
        dcaa: Length of sub-segment A (degcm)
        params: Dictionary containing:
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
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
                sab: lift at end of segment A (<lift units>)
                sbc: lift at end of segment B (<lift units>)
                dsdcaab: velocity at AB transition (<lift units>/degcm)
                dsdcabc: velocity at BC transition (<lift units>/degcm)
                d2sdca2ab: accel at AB transition (<lift units>/degcm^2)
    """
    # Initial conditions based on dcaa
    if dcaa > 0:
        params['dcaa'] = dcaa
        params['d2sdca2ab'] = params['amx']
    else:
        params['dcaa'] = 0
        params['d2sdca2ab'] = params['amx'] + dcaa * params['jmx']
    
    # Calculate segment A end conditions
    params['sab'] = params['amx']/2 * params['dcaa']**2 + params['S1']
    params['dsdcaab'] = params['amx'] * params['dcaa']
    
    # Calculate segment B length options - exactly as in MATLAB
    dcab1 = (params['dmx'] + params['d2sdca2ab'])/params['jmx']  # d2sdca2BC=-DMX
    dcab2 = (params['d2sdca2ab']/params['jmx'] + 
             np.sqrt((params['d2sdca2ab']/params['jmx'])**2 + 
                     2*params['dsdcaab']/params['jmx']))  # dsdcaBC=0
    
    # Select minimum length for segment B
    params['dcab'] = min(dcab1, dcab2)
    
    # Calculate segment B end conditions
    params['sbc'] = (-params['jmx']/6 * params['dcab']**3 + 
                     params['d2sdca2ab']/2 * params['dcab']**2 + 
                     params['dsdcaab'] * params['dcab'] + 
                     params['sab'])
    
    params['dsdcabc'] = (-params['jmx']/2 * params['dcab']**2 + 
                         params['d2sdca2ab'] * params['dcab'] + 
                         params['dsdcaab'])
    
    # Calculate segment C length
    params['dcac'] = max(0, params['dsdcabc']/params['dmx'])
    
    # Calculate lift error
    s2e = (-params['dmx']/2 * params['dcac']**2 + 
           params['dsdcabc'] * params['dcac'] + 
           params['sbc'])/params['S2'] - 1
    
    return s2e, params 