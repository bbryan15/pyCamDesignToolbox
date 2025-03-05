import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import warnings


def psvas2e(dcaa: float, params: dict) -> tuple[float, dict]:
    """
    Calculate lift error for p-sva (peak accel to specified lift, velocity, accel) segment.
    
    Args:
        dcaa: Length of segment A or negative for partial segment B with d2sdca21<Amx (degcm)
        params: Dictionary containing:
            S1: lift at start of segment (<lift units>)
            S2: lift at end of segment (<lift units>)
            vmatch: final velocity (>=0, <lift units>/degcm)
            amatch: final accel (>=0, <lift units>/degcm^2)
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
                sab: lift at end of segment A (<lift units>)
                sbc: lift at end of segment B (<lift units>)
                scd: lift at end of segment C (<lift units>)
                dsdcaab: velocity at AB transition (<lift units>/degcm)
                dsdcabc: velocity at BC transition (<lift units>/degcm)
                dsdcacd: velocity at CD transition (<lift units>/degcm)
                d2sdca2ab: accel at AB transition (<lift units>/degcm^2)
                d2sdca2cd: accel at CD transition (<lift units>/degcm^2)
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
    
    # Choose dcac to match final velocity to vmatch
    # dcac = length of segment C or negative for partial segment B with d2sdca2cd > -dmx (degcm)
    a = params['jmx']
    b = 2 * params['dmx']
    c = (-params['amx'] * params['dcaa'] - 
         ((params['d2sdca2ab']**2 + params['amatch']**2)/2 - params['dmx']**2)/params['jmx'] + 
         params['vmatch'])
    
    dcac = -c/params['dmx']  # dcac > 0 result
    
    if dcac > 0:
        params['dcac'] = dcac
        params['d2sdca2cd'] = -params['dmx']
    else:
        arg = b**2 - 4*a*c
        if arg < 0:
            warnings.warn("gcam0 function, 'p-sva' option: Unable to match v2 for current dcaa iteration. "
                        "Continuing with d2sdca2cd=0.")
        
        # dcac < 0 result: physical root for continuity with dcac>0 result
        # b^2-4*a*c=0 corresponds to min dcac=-dmx/jmx for d2sdca2cd = 0
        dcac = (-b + np.sqrt(max(arg, 0)))/(2*a)
        params['dcac'] = 0
        params['d2sdca2cd'] = -params['dmx'] - dcac * params['jmx']
    
    # Calculate remaining segment parameters
    params['dcab'] = (params['d2sdca2ab'] - params['d2sdca2cd'])/params['jmx']
    
    params['sbc'] = (-params['jmx']/6 * params['dcab']**3 + 
                     params['d2sdca2ab']/2 * params['dcab']**2 + 
                     params['dsdcaab'] * params['dcab'] + 
                     params['sab'])
    
    params['dsdcabc'] = (-params['jmx']/2 * params['dcab']**2 + 
                         params['d2sdca2ab'] * params['dcab'] + 
                         params['dsdcaab'])
    
    params['scd'] = (-params['dmx']/2 * params['dcac']**2 + 
                     params['dsdcabc'] * params['dcac'] + 
                     params['sbc'])
    
    params['dsdcacd'] = -params['dmx'] * params['dcac'] + params['dsdcabc']
    
    params['dcad'] = (params['amatch'] - params['d2sdca2cd'])/params['jmx']
    
    # Calculate lift error
    s2e = (params['jmx']/6 * params['dcad']**3 + 
           params['d2sdca2cd']/2 * params['dcad']**2 + 
           params['dsdcacd'] * params['dcad'] + 
           params['scd'])/params['S2'] - 1
    
    return s2e, params

def psvaca2e(J: float, params: dict) -> tuple[float, dict]:
    """
    Calculate cam angle error for p-sva (peak accel to specified lift, velocity, accel) segment.
    
    Args:
        J: jerk (<lift units>/degcm^3)
        params: Dictionary containing:
            ca1: Initial cam angle (degcm)
            ca2t: Target cam angle at end of segment (degcm)
            dca: Cam angle step (degcm)
            dcaa: Length of sub-segment A (degcm)
            dcab: Length of sub-segment B (degcm)
            dcac: Length of sub-segment C (degcm)
            dcad: Length of sub-segment D (degcm)
    
    Returns:
        tuple: (ca2e, params)
            ca2e: cam angle error at end of segment (degcm)
            params: Updated dictionary with calculated parameters
    """
    # Store J for use in psvas2e
    params['jmx'] = J
    
    # Initialize dcaa
    dcaa = 0
    
    # Get initial s2err
    s2err, params = psvas2e(dcaa, params)
    
    if s2err > 0:
        dcaamx = dcaa
        dcaamn = dcaa
        while s2err >= 0:
            dcaamn = dcaamn - params['dca']
            s2err, params = psvas2e(dcaamn, params)
        
        # Find zero crossing using scipy's root_scalar
        sol = root_scalar(lambda x: psvas2e(x, params)[0],
                         bracket=[dcaamn, dcaamx],
                         method='brentq')
        dcaa = sol.root
        _, params = psvas2e(dcaa, params)
        
    elif s2err < 0:
        dcaamn = dcaa
        dcaamx = dcaa
        while s2err <= 0:
            dcaamx = dcaamx + params['dca']
            s2err, params = psvas2e(dcaamx, params)
        
        # Find zero crossing
        sol = root_scalar(lambda x: psvas2e(x, params)[0],
                         bracket=[dcaamn, dcaamx],
                         method='brentq')
        dcaa = sol.root
        _, params = psvas2e(dcaa, params)
    
    # Calculate final cam angle error
    ca2e = params['ca1'] + params['dcaa'] + params['dcab'] + params['dcac'] + params['dcad'] - params['ca2t']
    
    return ca2e, params

def gcam0_p_sva(ca1: float, dca: float, s1: float, s2: float, 
              Vr: float, Vmatch: float, Amatch: float,
              Amx: float, Dmx: float, Jmx: float) -> pd.DataFrame:
    """
    Generate cam profile for p-sva (peak accel to specified lift, velocity, accel) segment.
    
    Args:
        ca1: Initial cam angle (degcm)
        dca: Cam angle step (degcm)
        s1: Initial lift (<lift units>)
        s2: Final lift (<lift units>)
        Vr: Ramp velocity (>=0, <lift units>/degcm)
        Vmatch: Match velocity (<lift units>/degcm)
        Amatch: Match acceleration (<lift units>/degcm^2)
        Amx: Max acceleration (<lift units>/degcm^2)
        Dmx: Max deceleration (>0, <lift units>/degcm^2)
        Jmx: Max jerk (<lift units>/degcm^3)
    
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
    if s2 > s1:  # rise segment
        # Iterate J<Jmx so that end of segment is an even multiple of dca
        params['ca2t'] = 0
        ca2Jmx, _ = psvaca2e(Jmx, params)
        rem = ca2Jmx % dca
        
        if rem == 0:
            J = Jmx
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + dca
            # Find first f where psvaca2e is positive
            for f in np.arange(0.9, 0, -0.1):
                if psvaca2e(f * Jmx, params)[0] > 0:
                    break
            
            # Find zero crossing
            sol = root_scalar(lambda x: psvaca2e(x, params)[0],
                            x0=f * Jmx,
                            x1=Jmx,
                            method='secant')
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = psvaca2e(J, params)
        
        if ca2 - ca1 >= 360:
            raise ValueError('required cam angle range >= 360')
        
        # Calculate transition points
        caAB = ca1 + params['dcaa']
        caBC = caAB + params['dcab']
        caCD = caBC + params['dcac']
        
        # Calculate csvaj matrix
        n = round((ca2 - ca1) / dca) + 1
        csvaj = np.zeros((n, 5))
        
        for i in range(n):
            csvaj[i, 0] = ca1 + dca * i
            
            if csvaj[i, 0] <= caAB and params['dcaa'] > 0:  # optional sub-segment A (rise): accel=Amx
                carel = csvaj[i, 0] - ca1
                csvaj[i, 1] = Amx/2 * carel**2 + s1
                csvaj[i, 2] = Amx * carel
                csvaj[i, 3] = Amx
                csvaj[i, 4] = 0
                
            elif csvaj[i, 0] <= caBC:  # sub-segment B (rise): jerk=-J
                carel = csvaj[i, 0] - caAB
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2ab']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                csvaj[i, 2] = -J/2 * carel**2 + params['d2sdca2ab'] * carel + params['dsdcaab']
                csvaj[i, 3] = -J * carel + params['d2sdca2ab']
                csvaj[i, 4] = -J
                
            elif csvaj[i, 0] <= caCD:  # optional sub-segment C (rise): accel=-Dmx
                carel = csvaj[i, 0] - caBC
                csvaj[i, 1] = -Dmx/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -Dmx * carel + params['dsdcabc']
                csvaj[i, 3] = -Dmx
                csvaj[i, 4] = 0
                
            else:  # optional sub-segment D (rise): jerk=J
                carel = csvaj[i, 0] - caCD
                csvaj[i, 1] = J/6 * carel**3 + params['d2sdca2cd']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = J/2 * carel**2 + params['d2sdca2cd'] * carel + params['dsdcacd']
                csvaj[i, 3] = J * carel + params['d2sdca2cd']
                csvaj[i, 4] = J
                
    else:  # fall segment: calculate from rise segment and symmetry
        # Swap s1 and s2 for fall calculation
        params['S1'], params['S2'] = s2, s1
        
        # Iterate J<Jmx so that end of segment is an even multiple of dca
        params['ca2t'] = 0
        ca2Jmx, _ = psvaca2e(Jmx, params)
        rem = ca2Jmx % dca
        
        if rem == 0:
            J = Jmx
            ca2 = ca2Jmx
        else:
            params['ca2t'] = ca2Jmx - rem + dca
            for f in np.arange(0.9, 0, -0.1):
                if psvaca2e(f * Jmx, params)[0] > 0:
                    break
            
            sol = root_scalar(lambda x: psvaca2e(x, params)[0],
                            x0=f * Jmx,
                            x1=Jmx,
                            method='secant')
            J = sol.root
            
            params['ca2t'] = 0
            ca2, params = psvaca2e(J, params)
        
        if ca2 - ca1 >= 360:
            raise ValueError('required cam angle range >= 360')
        
        # Calculate transition points
        caAB = ca2 - params['dcaa']
        caBC = caAB - params['dcab']
        caCD = caBC - params['dcac']
        
        # Calculate csvaj matrix
        n = round((ca2 - ca1) / dca) + 1
        csvaj = np.zeros((n, 5))
        
        for i in range(n-1, -1, -1):  # Reverse loop for fall segment
            csvaj[i, 0] = ca1 + dca * i
            
            if csvaj[i, 0] >= caAB and params['dcaa'] > 0:  # optional sub-segment A (fall): accel=Amx
                carel = ca2 - csvaj[i, 0]
                csvaj[i, 1] = Amx/2 * carel**2 + params['S1']
                csvaj[i, 2] = -Amx * carel
                csvaj[i, 3] = Amx
                csvaj[i, 4] = 0
                
            elif csvaj[i, 0] >= caBC:  # sub-segment B (fall): jerk=J
                carel = caAB - csvaj[i, 0]
                csvaj[i, 1] = -J/6 * carel**3 + params['d2sdca2ab']/2 * carel**2 + params['dsdcaab'] * carel + params['sab']
                csvaj[i, 2] = -(-J/2 * carel**2 + params['d2sdca2ab'] * carel + params['dsdcaab'])
                csvaj[i, 3] = -J * carel + params['d2sdca2ab']
                csvaj[i, 4] = J
                
            elif csvaj[i, 0] >= caCD:  # optional sub-segment C (fall): accel=-Dmx
                carel = caBC - csvaj[i, 0]
                csvaj[i, 1] = -Dmx/2 * carel**2 + params['dsdcabc'] * carel + params['sbc']
                csvaj[i, 2] = -(-Dmx * carel + params['dsdcabc'])
                csvaj[i, 3] = -Dmx
                csvaj[i, 4] = 0
                
            else:  # optional sub-segment D (fall): jerk=-J
                carel = caCD - csvaj[i, 0]
                csvaj[i, 1] = J/6 * carel**3 + params['d2sdca2cd']/2 * carel**2 + params['dsdcacd'] * carel + params['scd']
                csvaj[i, 2] = -(J/2 * carel**2 + params['d2sdca2cd'] * carel + params['dsdcacd'])
                csvaj[i, 3] = J * carel + params['d2sdca2cd']
                csvaj[i, 4] = -J
    
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 