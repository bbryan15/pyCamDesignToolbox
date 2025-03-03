import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


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

def gcam0_p_p(ca1: float, dca: float, s1: float, s2: float, 
               Vr: float, Vmatch: float, Amatch: float,
               Amx: float, Dmx: float, Jmx: float) -> pd.DataFrame:
    """
    Generate cam profile for p-p (peak accel to peak decel) segment.
    
    Returns:
        pd.DataFrame: csvaj matrix with columns ['ca', 's', 'v', 'a', 'j']
            ca: cam angle (degcm)
            s: lift (<lift units>)
            v: velocity (<lift units>/degcm)
            a: acceleration (<lift units>/degcm^2)
            j: jerk (<lift units>/degcm^3)
    """
    # Create params dictionary
    params = {
        'ca1': ca1, 'dca': dca, 'S1': s1, 'S2': s2,
        'vr': Vr, 'vmatch': Vmatch, 'amatch': Amatch,
        'amx': Amx, 'dmx': Dmx, 'jmx': Jmx,
        'ca2t': 0
    }

    # Handle rise or fall segment
    # fill in ...
        
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 