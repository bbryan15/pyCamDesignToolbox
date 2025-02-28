import pandas as pd
import numpy as np
from functions.conaccel import conaccel

def test_conaccel():
    print("\n=== Starting conaccel function test ===\n")
    
    # Test case parameters
    ca1, ca2 = 100, 150
    dca = 0.5
    s1, s2 = 0, 3.25
    v1, v2 = 0.04, 0.09
    ac = 0.001
    
    # Run all three modes
    modes = ['s2v2', 's2ac', 'v2ac']
    
    try:
        # Load MATLAB reference results
        matlab_result = pd.read_csv('test_data/matlab_conaccel.csv')
        
        for mode in modes:
            # Generate results from our Python function
            python_result = conaccel(ca1, ca2, dca, s1, s2, v1, v2, ac, mode)
            
            # Compare results
            max_diff = np.max(np.abs(python_result - matlab_result))
            print(f"\nTesting {mode} mode:")
            print(f"Maximum difference between MATLAB and Python results: {max_diff:.10f}")
            
            # Check if results match within tolerance
            tolerance = 1e-6
            if max_diff < tolerance:
                print(f"✓ {mode} test passed!")
            else:
                print(f"✗ {mode} test failed!")
                # Print where the differences occur
                diff_mask = np.abs(python_result - matlab_result) > tolerance
                print("\nDifferences found in rows:")
                print(python_result[diff_mask])
                print("\nExpected values (MATLAB):")
                print(matlab_result[diff_mask])
                
    except FileNotFoundError:
        print("\nWarning: MATLAB test data file not found")
        print("Python results for each mode:")
        for mode in modes:
            print(f"\n{mode} mode:")
            result = conaccel(ca1, ca2, dca, s1, s2, v1, v2, ac, mode)
            print(result)
    
    print("\n=== conaccel function test completed ===\n")

if __name__ == "__main__":
    test_conaccel() 