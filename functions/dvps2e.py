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
    # Case 1: d2sdca2CD = -DMX (may have segment D)
    dcac1 = (params['dmx'] + params['d2sdca2bc'])/params['jmx']
    # Case 2: dsdcaCD = 0 (no segment D)
    dcac2 = (params['d2sdca2bc']/params['jmx'] + 
             ((params['d2sdca2bc']/params['jmx'])**2 + 
              2*params['dsdcabc']/params['jmx'])**0.5)
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