import numpy as np
import pandas as pd
from pathlib import Path
from functions.gcam0 import gcam0

def test_gcam0_dvp():
    """
    Test function for gcam0 with dv-p case.
    Compares output with MATLAB results stored in CSV file.
    """
    # Test case parameters
    ca1, dca = 100, 0.5
    s1, s2 = 2, 2.5
    Vr = 0.05
    Vmatch = 0.05
    Amatch = 0.005
    Amx = 0.005
    Dmx = 0.005
    Jmx = 0.005
    prftype = 'dv-p'

    # Get result from gcam0
    result = gcam0(ca1, dca, s1, s2, Vr, Vmatch, Amatch, Amx, Dmx, Jmx, prftype)

    # Load MATLAB results from CSV
    csv_path = Path('test/matlab_gcam0_dvp.csv')
    matlab_df = pd.read_csv(csv_path)

    # Ensure both dataframes have the same columns
    assert all(result.columns == matlab_df.columns), "Column names don't match"
    
    # Compare shapes
    assert result.shape == matlab_df.shape, f"Shape mismatch: Python {result.shape} vs MATLAB {matlab_df.shape}"
    
    # Compare all values with tolerance
    tol = 1e-10
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        assert max_diff < tol, f"Maximum difference in {col}: {max_diff}"
    
    print("All dv-p tests passed successfully!")
    
    # Print maximum differences for each column
    print("\nMaximum absolute differences:")
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        print(f"{col}: {max_diff:.16f}")
    
    return result, matlab_df

def test_gcam0_dvd():
    """
    Test function for gcam0 with dv-d case.
    Compares output with MATLAB results stored in CSV file.
    """
    # Test case parameters
    ca1, dca = 100, 0.5
    s1, s2 = 2, 2.5
    Vr = 0.05
    Vmatch = 0.05
    Amatch = 0.005
    Amx = 0.005
    Dmx = 0.005
    Jmx = 0.005
    prftype = 'dv-d'

    # Get result from gcam0
    result = gcam0(ca1, dca, s1, s2, Vr, Vmatch, Amatch, Amx, Dmx, Jmx, prftype)

    # Load MATLAB results from CSV
    csv_path = Path('test/matlab_gcam0_dvd.csv')
    matlab_df = pd.read_csv(csv_path)

    # Ensure both dataframes have the same columns
    assert all(result.columns == matlab_df.columns), "Column names don't match"
    
    # Compare shapes
    assert result.shape == matlab_df.shape, f"Shape mismatch: Python {result.shape} vs MATLAB {matlab_df.shape}"
    
    # Compare all values with tolerance
    tol = 1e-10
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        assert max_diff < tol, f"Maximum difference in {col}: {max_diff}"
    
    print("All dv-d tests passed successfully!")
    
    # Print maximum differences for each column
    print("\nMaximum absolute differences:")
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        print(f"{col}: {max_diff:.16f}")
    
    return result, matlab_df

def test_gcam0_pp():
    """
    Test function for gcam0 with p-p case.
    Compares output with MATLAB results stored in CSV file.
    """
    # Test case parameters
    ca1, dca = 100, 0.5
    s1, s2 = 2, 2.5
    Vr = 0.05
    Vmatch = 0.05
    Amatch = 0.005
    Amx = 0.005
    Dmx = 0.005
    Jmx = 0.005
    prftype = 'p-p'

    # Get result from gcam0
    result = gcam0(ca1, dca, s1, s2, Vr, Vmatch, Amatch, Amx, Dmx, Jmx, prftype)

    # Load MATLAB results from CSV
    csv_path = Path('test/matlab_gcam0_pp.csv')
    matlab_df = pd.read_csv(csv_path)

    # Ensure both dataframes have the same columns
    assert all(result.columns == matlab_df.columns), "Column names don't match"
    
    # Compare shapes
    assert result.shape == matlab_df.shape, f"Shape mismatch: Python {result.shape} vs MATLAB {matlab_df.shape}"
    
    # Compare all values with tolerance
    tol = 1e-10
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        assert max_diff < tol, f"Maximum difference in {col}: {max_diff}"
    
    print("All p-p tests passed successfully!")
    
    # Print maximum differences for each column
    print("\nMaximum absolute differences:")
    for col in ['s', 'v', 'a', 'j']:
        max_diff = (result[col] - matlab_df[col]).abs().max()
        print(f"{col}: {max_diff:.16f}")
    
    return result, matlab_df

if __name__ == "__main__":
    # Run all tests
    print("Testing dv-p case:")
    result_dvp, matlab_dvp = test_gcam0_dvp()
    
    print("\n" + "="*80 + "\n")
    
    print("Testing dv-d case:")
    result_dvd, matlab_dvd = test_gcam0_dvd()
    
    print("\n" + "="*80 + "\n")
    
    print("Testing p-p case:")
    result_pp, matlab_pp = test_gcam0_pp()
    
    # Print detailed comparison for p-p case
    print("\nFirst few rows comparison (p-p):")
    print("\nPython result:")
    print(result_pp.head().to_string(float_format=lambda x: '{:.16f}'.format(x)))
    print("\nMATLAB result:")
    print(matlab_pp.head().to_string(float_format=lambda x: '{:.16f}'.format(x)))
    
    # Print middle rows (transition phase)
    print("\nMiddle rows comparison (p-p):")
    mid_point = len(result_pp) // 2
    print("\nPython result:")
    print(result_pp.iloc[mid_point-2:mid_point+3].to_string(float_format=lambda x: '{:.16f}'.format(x)))
    print("\nMATLAB result:")
    print(matlab_pp.iloc[mid_point-2:mid_point+3].to_string(float_format=lambda x: '{:.16f}'.format(x)))
    
    # Print last few rows
    print("\nLast few rows comparison (p-p):")
    print("\nPython result:")
    print(result_pp.tail().to_string(float_format=lambda x: '{:.16f}'.format(x)))
    print("\nMATLAB result:")
    print(matlab_pp.tail().to_string(float_format=lambda x: '{:.16f}'.format(x))) 