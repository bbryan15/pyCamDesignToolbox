import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

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

def gcam0_d_v(ca1: float, dca: float, s1: float, s2: float, 
               Vr: float, Vmatch: float, Amatch: float,
               Amx: float, Dmx: float, Jmx: float) -> pd.DataFrame:
    """
    Generate cam profile for d-v (dwell to velocity) segment.
    
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

    if params['S2'] > params['S1']:  # rise segment
            # Iterate J<Jmx so that end of segment is an even multiple of dca
            params['ca2t'] = 0
            ca2Jmx, _ = dvca2e(params['jmx'], params)
            rem = ca2Jmx % params['dca']
            
            if rem == 0:
                J = params['jmx']
                ca2 = ca2Jmx
            else:
                params['ca2t'] = ca2Jmx - rem + params['dca']
                # Find first f where dvca2e is positive
                for f in np.arange(0.9, 0, -0.1):
                    if dvca2e(f * params['jmx'], params)[0] > 0:
                        break
                
                # Direct equivalent to MATLAB's fzero
                sol = root_scalar(lambda x: dvca2e(x, params)[0],
                                x0=f * params['jmx'],
                                x1=params['jmx'],
                                method='secant')
                J = sol.root
                
                params['ca2t'] = 0
                ca2, params = dvca2e(J, params)
            
            if ca2 - params['ca1'] >= 360:
                raise ValueError('required cam angle range >= 360')
            
            # Calculate transition points
            caAB = params['ca1'] + params['dcaa']
            caBC = caAB + params['dcab']
            caCD = caBC + params['dcac']
            
            # Calculate csvaj matrix
            n = round((ca2 - params['ca1']) / params['dca']) + 1
            csvaj = np.zeros((n, 5))  # pre-allocate for compute speed
            
            for i in range(n):
                csvaj[i, 0] = params['ca1'] + params['dca'] * i
                
                if csvaj[i, 0] <= caAB:  # sub-segment A (rise): jerk=J
                    carel = csvaj[i, 0] - params['ca1']
                    csvaj[i, 1] = J/6 * carel**3 + params['S1']
                    csvaj[i, 2] = J/2 * carel**2
                    csvaj[i, 3] = J * carel
                    csvaj[i, 4] = J
                    
                elif csvaj[i, 0] <= caBC and params['dcab'] > 0:  # optional sub-segment B (rise): accel=Amx
                    carel = csvaj[i, 0] - caAB
                    csvaj[i, 1] = params['amx']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                    csvaj[i, 2] = params['amx'] * carel + params['dsdcaab']
                    csvaj[i, 3] = params['amx']
                    csvaj[i, 4] = 0
                    
                elif csvaj[i, 0] <= caCD:  # sub-segment C (rise): jerk=-J
                    carel = csvaj[i, 0] - caBC
                    csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                    csvaj[i, 2] = -J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc']
                    csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                    csvaj[i, 4] = -J
                    
                else:  # sub-segment D (rise): vel=Vr
                    carel = csvaj[i, 0] - caCD
                    csvaj[i, 1] = params['vr'] * carel + params['scd']
                    csvaj[i, 2] = params['vr']
                    csvaj[i, 3] = 0
                    csvaj[i, 4] = 0
                    
    else:  # fall segment: calculate from rise segment symmetry
            # Swap S1 and S2 for fall calculation
            params['S1'], params['S2'] = params['S2'], params['S1']
            
            # Iterate J<Jmx so that end of segment is an even multiple of dca
            params['ca2t'] = 0
            ca2Jmx, _ = dvca2e(params['jmx'], params)
            rem = ca2Jmx % params['dca']
            
            if rem == 0:
                J = params['jmx']
                ca2 = ca2Jmx
            else:
                params['ca2t'] = ca2Jmx - rem + params['dca']
                for f in np.arange(0.9, 0, -0.1):
                    if dvca2e(f * params['jmx'], params)[0] > 0:
                        break
                
                sol = root_scalar(lambda x: dvca2e(x, params)[0],
                                x0=f * params['jmx'],
                                x1=params['jmx'],
                                method='secant')
                J = sol.root
                
                params['ca2t'] = 0
                ca2, params = dvca2e(J, params)
                
            if ca2 - params['ca1'] >= 360:
                raise ValueError('required cam angle range >= 360')
                
            # Calculate transition points
            caAB = ca2 - params['dcaa']
            caBC = caAB - params['dcab']
            caCD = caBC - params['dcac']
            
            # Calculate csvaj matrix
            n = round((ca2 - params['ca1']) / params['dca']) + 1
            csvaj = np.zeros((n, 5))  # pre-allocate for compute speed
            
            for i in range(n-1, -1, -1):  # Reverse loop for fall segment
                csvaj[i, 0] = params['ca1'] + params['dca'] * i
                
                if csvaj[i, 0] >= caAB:  # sub-segment A (fall): jerk=-J
                    carel = ca2 - csvaj[i, 0]
                    csvaj[i, 1] = J/6 * carel**3 + params['S1']
                    csvaj[i, 2] = -J/2 * carel**2
                    csvaj[i, 3] = J * carel
                    csvaj[i, 4] = -J
                    
                elif csvaj[i, 0] >= caBC and params['dcab'] > 0:  # optional sub-segment B (fall): accel=Amx
                    carel = caAB - csvaj[i, 0]
                    csvaj[i, 1] = params['amx']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                    csvaj[i, 2] = -(params['amx'] * carel + params['dsdcaab'])
                    csvaj[i, 3] = params['amx']
                    csvaj[i, 4] = 0
                    
                elif csvaj[i, 0] >= caCD:  # sub-segment C (fall): jerk=J
                    carel = caBC - csvaj[i, 0]
                    csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                    csvaj[i, 2] = -(-J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc'])
                    csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                    csvaj[i, 4] = J
                    
                else:  # sub-segment D (fall): vel=-Vr
                    carel = caCD - csvaj[i, 0]
                    csvaj[i, 1] = params['vr'] * carel + params['scd']
                    csvaj[i, 2] = -params['vr']
                    csvaj[i, 3] = 0
                    csvaj[i, 4] = 0    
        
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 