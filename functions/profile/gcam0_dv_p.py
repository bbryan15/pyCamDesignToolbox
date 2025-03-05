import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


def dvps2e(cacbr: float, params: dict) -> tuple[float, dict]:
    """
    Calculate lift error for dv-p case (dwell or const velocity to peak)
    
    Args:
        cacbr: relative cam angle vs ca1 at which the accel is cut back (degcm)
        params: dictionary containing:
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
            vr: velocity at start or end of segment (>=0, <lift units>/degcm)
            amx: max accel (<lift units>/degcm^2)
            dmx: max decel (>0, <lift units>/degcm^2)
            jmx: max jerk (<lift units>/degcm^3)
    
    Returns:
        tuple: (s2e, updated_params)
            s2e: lift error at end of segment (%)
            updated_params: dictionary with updated values for:
                dcaa: length of sub-segment A (degcm)
                dcab: length of sub-segment B (degcm)
                dcac: length of sub-segment C (degcm)
                dcad: length of sub-segment D (degcm)
                sab: lift at end of segment A
                sbc: lift at end of segment B
                scd: lift at end of segment C
                dsdcaab: velocity at end of segment A
                dsdcabc: velocity at end of segment B
                dsdcacd: velocity at end of segment C
                d2sdca2bc: accel at end of segment B
    """
    # Check if cacbr exceeds AMX/JMX ratio
    if cacbr > params['amx']/params['jmx']:
        params['dcaa'] = params['amx']/params['jmx']
        params['dcab'] = cacbr - params['dcaa']
        params['d2sdca2bc'] = params['amx']
    else:
        params['dcaa'] = cacbr
        params['dcab'] = 0
        params['d2sdca2bc'] = params['jmx'] * cacbr

    # Calculate segment A end conditions
    params['sab'] = (params['jmx']/6 * params['dcaa']**3 + 
                    params['vr'] * params['dcaa'] + params['S1'])
    params['dsdcaab'] = params['jmx']/2 * params['dcaa']**2 + params['vr']

    # Calculate segment B end conditions
    params['sbc'] = (params['amx']/2 * params['dcab']**2 + 
                    params['dsdcaab'] * params['dcab'] + params['sab'])
    params['dsdcabc'] = params['amx'] * params['dcab'] + params['dsdcaab']

    # Calculate segment C length (minimum of two possible values)
    dcac1 = (params['dmx'] + params['d2sdca2bc'])/params['jmx']  # d2sdca2CD = -DMX
    dcac2 = (params['d2sdca2bc']/params['jmx'] + 
             ((params['d2sdca2bc']/params['jmx'])**2 + 
              2*params['dsdcabc']/params['jmx'])**0.5)  # dsdcaCD = 0
    params['dcac'] = min(dcac1, dcac2)

    # Calculate segment C end conditions
    params['scd'] = (-params['jmx']/6 * params['dcac']**3 + 
                    params['d2sdca2bc']/2 * params['dcac']**2 + 
                    params['dsdcabc'] * params['dcac'] + params['sbc'])
    params['dsdcacd'] = (-params['jmx']/2 * params['dcac']**2 + 
                        params['d2sdca2bc'] * params['dcac'] + 
                        params['dsdcabc'])

    # Calculate segment D length
    params['dcad'] = max(0, params['dsdcacd']/params['dmx'])

    # Calculate lift error as percentage
    s2e = ((-params['dmx']/2 * params['dcad']**2 + 
            params['dsdcacd'] * params['dcad'] + 
            params['scd'])/params['S2'] - 1)

    return s2e, params

def dvpca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for dv-p segment.
    
    Args:
        J: max jerk (lift units/degcm^3)
        params: dictionary containing:
            ca1: cam angle at start of segment (degcm)
            ca2t: target cam angle at end of segment (degcm)
            dca: cam angle step (degcm)
            amx: max acceleration
            dcaa, dcab, dcac, dcad: segment lengths
    
    Returns:
        tuple: (ca2e, updated_params)
            ca2e: cam angle error at end of segment
            updated_params: dictionary with updated values
    """
    # Store jerk value
    params['jmx'] = J
    
    # Calculate ratio of max accel to jerk
    cacbr = params['amx'] / J
    
    # Calculate lift error
    s2err, params = dvps2e(cacbr, params)
    
    if s2err > 0:
        # Error is positive, search downward
        cacbrmx = cacbr
        cacbrmn = cacbr
        while s2err >= 0:
            cacbrmn = cacbrmn - params['dca']
            s2err, params = dvps2e(cacbrmn, params)
        # Find zero crossing using root_scalar
        sol = root_scalar(lambda x: dvps2e(x, params.copy())[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        _, params = dvps2e(cacbr, params)
        
    elif s2err < 0:
        # Error is negative, search upward
        cacbrmn = cacbr
        cacbrmx = cacbr
        while s2err <= 0:
            cacbrmx = cacbrmx + params['dca']
            s2err, params = dvps2e(cacbrmx, params)
        # Find zero crossing
        sol = root_scalar(lambda x: dvps2e(x, params.copy())[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        _, params = dvps2e(cacbr, params)
    
    # Calculate final cam angle error
    ca2e = (params['ca1'] + params['dcaa'] + params['dcab'] + 
            params['dcac'] + params['dcad'] - params['ca2t'])
    
    return ca2e, params

def gcam0_dv_p(ca1: float, dca: float, s1: float, s2: float, 
               Vr: float, Vmatch: float, Amatch: float,
               Amx: float, Dmx: float, Jmx: float) -> pd.DataFrame:
    """
    Generate cam profile for dv-p (dwell/const vel to peak decel) segment.
    
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
                            x0=f * params['jmx'],
                            x1=params['jmx'],
                            method='secant')
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
        csvaj = np.zeros((n, 5))
        
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
            for f in np.arange(0.9, 0, -0.1):
                if dvpca2e(f * params['jmx'], params)[0] > 0:
                    break
            
            sol = root_scalar(lambda x: dvpca2e(x, params)[0],
                            x0=f * params['jmx'],
                            x1=params['jmx'],
                            method='secant')
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
        csvaj = np.zeros((n, 5))
        
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
        
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 