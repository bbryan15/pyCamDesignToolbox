import numpy as np
from scipy.optimize import root_scalar
from .dvpca2e import dvpca2e

def gcam0_dvp(params: dict) -> np.ndarray:
    """
    Generate cam profile for dv-p (dwell or constant velocity to peak) segment.
    Peak velocity must be greater than starting velocity.
    
    Args:
        params: Dictionary containing all necessary parameters
            ca1, dca, S1, S2, vr, amx, dmx, etc.
    
    Returns:
        np.ndarray: csvaj matrix with columns [ca, s, v, a, j]
    """
    # Handle rise or fall segment
    if params['S2'] > params['S1']:  # rise segment
        # Iterate J<Jmx so that end of segment is an even multiple of dca
        params['ca2t'] = 0
        ca2Jmx, _ = dvpca2e(params['jmx'], params)
        rem = ca2Jmx % params['dca']
        
        if rem == 0:
            J = params['jmx']
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + params['dca']
            # Find first f where dvpca2e is positive
            for f in np.arange(0.9, 0, -0.1):
                if dvpca2e(f * params['jmx'], params)[0] > 0:
                    break
            
            # Direct equivalent to MATLAB's fzero
            sol = root_scalar(lambda x: dvpca2e(x, params)[0],
                            x0=f * params['jmx'],  # Initial guess
                            x1=params['jmx'],      # Second guess
                            method='secant')       # Using secant method instead of Brentq
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = dvpca2e(J, params)
        
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
                csvaj[i, 1] = J/6 * carel**3 + params['vr'] * carel + params['S1']
                csvaj[i, 2] = J/2 * carel**2 + params['vr']
                csvaj[i, 3] = J * carel
                csvaj[i, 4] = J
                
            elif csvaj[i, 0] <= caBC and params['dcab'] > 0:  # optional sub-segment B (rise): accel=Amx
                carel = csvaj[i, 0] - caAB
                csvaj[i, 1] = params['amx']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                csvaj[i, 2] = params['amx'] * carel + params['dsdcaab']
                csvaj[i, 3] = params['amx']
                csvaj[i, 4] = 0
                
            elif csvaj[i, 0] <= caCD or params['dcac'] == 0:  # sub-segment C (rise): jerk=-J
                carel = csvaj[i, 0] - caBC
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc']
                csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                csvaj[i, 4] = -J
                
            else:  # optional sub-segment D (rise): accel=-Dmx
                carel = csvaj[i, 0] - caCD
                csvaj[i, 1] = -params['dmx']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = -params['dmx'] * carel + params['dsdcacd']
                csvaj[i, 3] = -params['dmx']
                csvaj[i, 4] = 0
                
    else:  # fall segment: calculate from rise segment symmetry
        # Store original S1, S2 values
        orig_s1, orig_s2 = params['S1'], params['S2']
        
        # Swap S1 and S2 for fall calculation
        params['S1'] = orig_s2
        params['S2'] = orig_s1
        
        # Iterate J<Jmx so that end of segment is an even multiple of dca
        params['ca2t'] = 0
        ca2Jmx, _ = dvpca2e(params['jmx'], params)
        rem = ca2Jmx % params['dca']
        
        if rem == 0:
            J = params['jmx']
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + params['dca']
            # Find first f where dvpca2e is positive
            for f in np.arange(0.9, 0, -0.1):
                if dvpca2e(f * params['jmx'], params)[0] > 0:
                    break
            
            # Direct equivalent to MATLAB's fzero
            sol = root_scalar(lambda x: dvpca2e(x, params)[0],
                            x0=f * params['jmx'],  # Initial guess
                            x1=params['jmx'],      # Second guess
                            method='secant')       # Using secant method instead of Brentq
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = dvpca2e(J, params)
            
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
                csvaj[i, 1] = J/6 * carel**3 + params['vr'] * carel + params['S1']
                csvaj[i, 2] = -(J/2 * carel**2 + params['vr'])
                csvaj[i, 3] = J * carel
                csvaj[i, 4] = -J
                
            elif csvaj[i, 0] >= caBC and params['dcab'] > 0:  # optional sub-segment B (fall): accel=Amx
                carel = caAB - csvaj[i, 0]
                csvaj[i, 1] = params['amx']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                csvaj[i, 2] = -(params['amx'] * carel + params['dsdcaab'])
                csvaj[i, 3] = params['amx']
                csvaj[i, 4] = 0
                
            elif csvaj[i, 0] >= caCD or params['dcac'] == 0:  # sub-segment C (fall): jerk=J
                carel = caBC - csvaj[i, 0]
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -(-J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc'])
                csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                csvaj[i, 4] = J
                
            else:  # optional sub-segment D (fall): accel=-Dmx
                carel = caCD - csvaj[i, 0]
                csvaj[i, 1] = -params['dmx']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = -(-params['dmx'] * carel + params['dsdcacd'])
                csvaj[i, 3] = -params['dmx']
                csvaj[i, 4] = 0
        
    return csvaj 