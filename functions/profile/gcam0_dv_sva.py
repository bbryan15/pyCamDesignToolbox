import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


def dvsvas2e(cacbr: float, params: dict) -> tuple[float, dict]:
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

def dvsvaca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for dv-sva (dwell or constant velocity to specified velocity and acceleration) segment.
    
    Args:
        J: max jerk (lift units/degcm^3)
        params: Dictionary containing:
            ca1: cam angle at start of segment (degcm)
            ca2t: target cam angle at end of segment (degcm)
            dca: cam angle step (degcm): typically 0.5 or 1.0
            dcaa: length of sub-segment A (degcm)
            dcab: length of sub-segment B (degcm)
            dcac: length of sub-segment C (degcm)
            dcad: length of sub-segment D (degcm)
            dcae: length of sub-segment E (degcm)
            amx: max accel limit (<lift units>/degcm^2)
    
    Returns:
        tuple: (ca2e, params)
            ca2e: cam angle error at end of segment (degcm)
            params: Updated dictionary with calculated parameters
    """
    # Store J for use in dvsvas2e
    params['jmx'] = J
    
    # Calculate initial cacbr (cam angle change before reversal)
    cacbr = params['amx'] / J
    
    # Get initial s2err
    s2err, params = dvsvas2e(cacbr, params)
    
    if s2err > 0:
        cacbrmx = cacbr
        cacbrmn = cacbr
        while s2err >= 0:
            cacbrmn = cacbrmn - params['dca']
            s2err, params = dvsvas2e(cacbrmn, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: dvsvas2e(x, params)[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
        
    elif s2err < 0:
        cacbrmn = cacbr
        cacbrmx = cacbr
        while s2err <= 0:
            cacbrmx = cacbrmx + params['dca']
            s2err, params = dvsvas2e(cacbrmx, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: dvsvas2e(x, params)[0],
                         bracket=[cacbrmn, cacbrmx],
                         method='brentq')
        cacbr = sol.root
    
    # Calculate final cam angle error
    ca2e = (params['ca1'] + params['dcaa'] + params['dcab'] + 
            params['dcac'] + params['dcad'] + params['dcae'] - params['ca2t'])
    
    return ca2e, params 

def gcam0_dv_sva(ca1: float, dca: float, s1: float, s2: float, 
               Vr: float, Vmatch: float, Amatch: float,
               Amx: float, Dmx: float, Jmx: float) -> pd.DataFrame:
    """
    Generate cam profile for dv-sva (dwell or constant velocity to specified velocity and acceleration) segment.
    
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
        ca2Jmx, _ = dvsvaca2e(params['jmx'], params)
        rem = ca2Jmx % params['dca']
        
        if rem == 0:
            J = params['jmx']
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + params['dca']
            # Find first f where dvsvaca2e is positive
            for f in np.arange(0.9, 0, -0.1):
                if dvsvaca2e(f * params['jmx'], params)[0] > 0:
                    break
            
            # Direct equivalent to MATLAB's fzero
            sol = root_scalar(lambda x: dvsvaca2e(x, params)[0],
                            x0=f * params['jmx'],
                            x1=params['jmx'],
                            method='secant')
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = dvsvaca2e(J, params)
        
        if ca2 - params['ca1'] >= 360:
            raise ValueError('required cam angle range >= 360')
        
        # Calculate transition points
        caAB = params['ca1'] + params['dcaa']
        caBC = caAB + params['dcab']
        caCD = caBC + params['dcac']
        caDE = caCD + params['dcad']
        
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
                
            elif csvaj[i, 0] <= caCD:  # sub-segment C (rise): jerk=-J
                carel = csvaj[i, 0] - caBC
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc']
                csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                csvaj[i, 4] = -J
                
            elif csvaj[i, 0] <= caDE and params['dcad'] > 0:  # optional sub-segment D (rise): accel=-Dmx
                carel = csvaj[i, 0] - caCD
                csvaj[i, 1] = -params['dmx']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = -params['dmx'] * carel + params['dsdcacd']
                csvaj[i, 3] = -params['dmx']
                csvaj[i, 4] = 0
                
            else:  # optional sub-segment E (rise): jerk=J
                carel = csvaj[i, 0] - caDE
                csvaj[i, 1] = J/6 * carel**3 + params['d2sdca2de']/2 * carel**2 + params['dsdcade'] * carel + params['sde']
                csvaj[i, 2] = J/2 * carel**2 + params['d2sdca2de'] * carel + params['dsdcade']
                csvaj[i, 3] = J * carel + params['d2sdca2de']
                csvaj[i, 4] = J
                
    else:  # fall segment: calculate from rise segment symmetry
        # Swap S1 and S2 for fall calculation
        params['S1'], params['S2'] = params['S2'], params['S1']
        
        # Iterate J<Jmx so that end of segment is an even multiple of dca
        params['ca2t'] = 0
        ca2Jmx, _ = dvsvaca2e(params['jmx'], params)
        rem = ca2Jmx % params['dca']
        
        if rem == 0:
            J = params['jmx']
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + params['dca']
            for f in np.arange(0.9, 0, -0.1):
                if dvsvaca2e(f * params['jmx'], params)[0] > 0:
                    break
            
            sol = root_scalar(lambda x: dvsvaca2e(x, params)[0],
                            x0=f * params['jmx'],
                            x1=params['jmx'],
                            method='secant')
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = dvsvaca2e(J, params)
        
        if ca2 - params['ca1'] >= 360:
            raise ValueError('required cam angle range >= 360')
        
        # Calculate transition points for fall segment
        caAB = ca2 - params['dcaa']
        caBC = caAB - params['dcab']
        caCD = caBC - params['dcac']
        caDE = caCD - params['dcad']
        
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
                
            elif csvaj[i, 0] >= caCD:  # sub-segment C (fall): jerk=J
                carel = caBC - csvaj[i, 0]
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2bc']/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -(-J/2 * carel**2 + params['d2sdca2bc'] * carel + params['dsdcabc'])
                csvaj[i, 3] = -J * carel + params['d2sdca2bc']
                csvaj[i, 4] = J
                
            elif csvaj[i, 0] >= caDE and params['dcad'] > 0:  # optional sub-segment D (fall): accel=-Dmx
                carel = caCD - csvaj[i, 0]
                csvaj[i, 1] = -params['dmx']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = -(-params['dmx'] * carel + params['dsdcacd'])
                csvaj[i, 3] = -params['dmx']
                csvaj[i, 4] = 0
                
            else:  # optional sub-segment E (fall): jerk=-J
                carel = caDE - csvaj[i, 0]
                csvaj[i, 1] = J/6 * carel**3 + params['d2sdca2de']/2 * carel**2 + params['dsdcade'] * carel + params['sde']
                csvaj[i, 2] = -(J/2 * carel**2 + params['d2sdca2de'] * carel + params['dsdcade'])
                csvaj[i, 3] = J * carel + params['d2sdca2de']
                csvaj[i, 4] = -J
        
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 