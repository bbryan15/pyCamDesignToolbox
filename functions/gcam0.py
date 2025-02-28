import numpy as np
import pandas as pd
from scipy.optimize import fsolve, root_scalar
from typing import Dict, Tuple
from .gcam0_dvp import gcam0_dvp
from .gcam0_dv import gcam0_dv
from .gcam0_dvsva import gcam0_dvsva
from .gcam0_dvd import gcam0_dvd
from .gcam0_pp import gcam0_pp


def gcam0(ca1: float, dca: float, s1: float, s2: float, Vr: float, Vmatch: float, 
          Amatch: float, Amx: float, Dmx: float, Jmx: float, prftype: str) -> pd.DataFrame:
    """
    Generate cam profile for various motion types.
    
    Args:
        ca1: Initial cam angle (degcm)
        dca: Cam angle step (degcm): typically 0.5 or 1.0
        s1: Initial lift (<lift units>)
        s2: Final lift (<lift units>)
        Vr: Ramp velocity (>=0, <lift units>/degcm)
        Vmatch: Final velocity for sva profile types (>=0, <lift units>/degcm)
        Amatch: Final accel for sva profile types (>=0, <lift units>/degcm^2)
        Amx: Max accel limit (<lift units>/degcm^2)
        Dmx: Max decel limit (>0, <lift units>/degcm^2)
        Jmx: Max absolute jerk limit (<lift units>/degcm^3)
        prftype: Profile type ('dv-p', 'd-v', 'dv-sva', 'dv-d', 'p-p', etc.)
    
    Returns:
        np.ndarray: csvaj matrix with columns [ca, s, v, a, j]
    """
    params = {
        # Primary parameters (matching first global line)
        'ca1': ca1,        # Initial cam angle (degcm)
        'ca2t': 0,         # Target cam angle at end of segment (degcm)
        'dca': dca,        # Cam angle step (degcm): typically 0.5 or 1.0
        'S1': s1,          # Initial lift (<lift units>)
        'S2': s2,          # Final lift (<lift units>)
        'vr': Vr,          # Ramp velocity (>=0, <lift units>/degcm)
        'vmatch': Vmatch,  # Final velocity for sva profile types (>=0, <lift units>/degcm)
        'amatch': Amatch,  # Final accel for sva profile types (>=0, <lift units>/degcm^2)
        'amx': Amx,        # Max accel limit (<lift units>/degcm^2)
        'dmx': Dmx,        # Max decel limit (>0, <lift units>/degcm^2)
        'jmx': Jmx,        # Max absolute jerk limit (<lift units>/degcm^3)
        
        # Secondary parameters (matching second global line)
        'dcaa': 0,         # Length of sub-segment A (degcm)
        'dcab': 0,         # Length of sub-segment B (degcm)
        'dcac': 0,         # Length of sub-segment C (degcm)
        'dcad': 0,         # Length of sub-segment D (degcm)
        'dcae': 0,         # Length of sub-segment E (degcm)
        'sab': 0,          # Position at end of segment A / start of B
        'sbc': 0,          # Position at end of segment B / start of C
        'scd': 0,          # Position at end of segment C / start of D
        'sde': 0,          # Position at end of segment D / start of E
        'dsdcaab': 0,      # Velocity at AB transition
        'dsdcabc': 0,      # Velocity at BC transition
        'dsdcacd': 0,      # Velocity at CD transition
        'dsdcade': 0,      # Velocity at DE transition
        'd2sdca2ab': 0,    # Acceleration at AB transition
        'd2sdca2bc': 0,    # Acceleration at BC transition
        'd2sdca2cd': 0,    # Acceleration at CD transition
        'd2sdca2de': 0     # Acceleration at DE transition
    }
    
    # Profile types:
    # 'd-v'    : dwell to constant velocity
    # 'dv-p'   : dwell or constant velocity to peak decel, peak velocity > starting velocity
    # 'dv-d'   : dwell or constant velocity to dwell, peak velocity > starting velocity
    # 'p-p'    : peak accel to peak decel
    # 'p-d'    : peak accel to dwell
    # 'dv-sva' : dwell or constant velocity to specified lift, velocity, and accel
    # 'p-sva'  : peak accel to specified lift, velocity, and accel
    
    # Generate profile based on type
    if prftype == 'dv-p':
        # dv-p: dwell or constant velocity to peak decel
        # Peak velocity must be greater than starting velocity
        csvaj = gcam0_dvp(params)
    elif prftype == 'd-v':
        # d-v: dwell to constant velocity
        csvaj = gcam0_dv(params)
    elif prftype == 'dv-sva':
        # dv-sva: dwell or constant velocity to specified velocity and acceleration
        csvaj = gcam0_dvsva(params)
    elif prftype == 'dv-d':
        # dv-d: dwell or constant velocity to dwell
        csvaj = gcam0_dvd(params)
    elif prftype == 'p-p':
        # p-p: peak accel to peak decel
        csvaj = gcam0_pp(params)
    else:
        raise ValueError(f'Unknown profile type: {prftype}')
    
    return pd.DataFrame(csvaj, columns=['ca', 's', 'v', 'a', 'j']) 