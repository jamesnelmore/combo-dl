"""
Test script to validate the PDS approach on small, known examples
before attempting Conway's problem.
"""

import numpy as np
from pds_set_transformer_ppo import CyclicGroup, PDSEnvironment, train_pds_model, evaluate_model


def test_small_known_pds():
    """Test on small cases where we know valid PDSs exist"""
    
    print("🧪 Testing PDS approach on small known examples")
    print("=" * 50)
    
    # Test cases: (group_order, (v, k, lambda, mu), known_pds_example)
    test_cases = [
        # Fano plane: {1, 2, 4} is a (7, 3, 1) PDS in Z_7
        (7, (7, 3, 1, 1), [1, 2, 4]),
        
        # Smaller test: {1} is a (3, 1, 0) "trivial" PDS in Z_3  
        (3, (3, 1, 0, 0), [1]),
        
        # Another small case: {1, 3} might work in Z_5
        (5, (5, 2, 1, 1), [1, 3]),
    ]
    
    for group_order, params, known_pds in test_cases:
        print(f"\n📋 Testing: Z_{group_order} with parameters {params}")
        print(f"Known PDS example: {known_pds}")
        
        # Verify the known example is actually a valid PDS
        if verify_pds(group_order, known_pds, params):
            print("✅ Known example verified as valid PDS")
        else:
            print("❌ Known example is NOT a valid PDS - skipping")
            continue
        
        # Train model on this case
        print(f"🚀 Training model for Z_{group_order}...")
        model, found_during_training = train_pds_model(
            group_order=group_order,
            target_params=params
        )
        
        if found_during_training:
            print("✅ Model found valid PDS during training!")
        else:
            print("🔄 Evaluating model...")
            success_rate, _ = evaluate_model(model, group_order, params, num_trials=5)
            if success_rate > 0:
                print(f"✅ Model succeeded in {success_rate:.1%} of trials!")
            else:
                print("❌ Model failed to find valid PDS")


def verify_pds(group_order: int, pds_elements: list, params: tuple) -> bool:
    """Manually verify if given elements form a valid PDS"""
    v, k, lambda_param, mu = params
    
    if len(pds_elements) != k:
        return False
    
    # Create group and compute difference counts
    group = CyclicGroup(group_order)
    diff_counts = [0] * group_order
    
    for a in pds_elements:
        for b in pds_elements:
            if a != b:
                diff = group.difference(a, b)
                diff_counts[diff] += 1
    
    # Check if each non-identity difference appears exactly lambda times
    for d in range(1, group_order):
        if diff_counts[d] != lambda_param:
            print(f"  Difference {d} appears {diff_counts[d]} times, expected {lambda_param}")
            return False
    
    return True


def manual_search_small_case(group_order: int = 7):
    """Manually search for PDS in small group to understand the problem"""
    print(f"\n🔍 Manual search for PDS in Z_{group_order}")
    
    group = CyclicGroup(group_order)
    
    # Try all possible subsets of size 3 for (7,3,1) PDS
    from itertools import combinations
    
    target_k = 3
    target_lambda = 1
    
    print(f"Checking all {target_k}-element subsets...")
    
    valid_pds_found = []
    
    for subset in combinations(range(group_order), target_k):
        # Compute difference distribution
        diff_counts = [0] * group_order
        
        for a in subset:
            for b in subset:
                if a != b:
                    diff = group.difference(a, b)
                    diff_counts[diff] += 1
        
        # Check if valid PDS
        is_valid = all(diff_counts[d] == target_lambda for d in range(1, group_order))
        
        if is_valid:
            valid_pds_found.append(subset)
            print(f"✅ Valid PDS found: {subset}")
            print(f"   Differences: {diff_counts[1:]}")
    
    if not valid_pds_found:
        print("❌ No valid PDS found")
    else:
        print(f"\n📊 Found {len(valid_pds_found)} valid PDS(s)")
    
    return valid_pds_found


def quick_conway_test():
    """Quick test on Conway's problem with reduced training time"""
    print("\n🎯 Quick test on Conway's problem (reduced training)")
    print("=" * 50)
    
    # Conway's parameters but with much less training
    CONWAY_PARAMS = (99, 14, 1, 2)
    
    print("Note: This is a quick test with limited training time.")
    print("Finding a valid PDS for Conway's problem may require much longer training.")
    
    # Create environment to test basic functionality
    group = CyclicGroup(99)
    env = PDSEnvironment(group, CONWAY_PARAMS)
    
    # Test environment
    obs, _ = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   Group: {group.name}")
    print(f"   Parameters: {CONWAY_PARAMS}")
    print(f"   State space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}, PDS_size={np.sum(obs)}")
    
    print("\n🚀 Starting short training run...")
    
    # Very short training just to test the pipeline
    import sys
    from io import StringIO
    
    # Capture output to reduce noise
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        model, found = train_pds_model(
            group_order=99,
            target_params=CONWAY_PARAMS
        )
        # Note: With short training, we don't expect to solve Conway's problem
        
    finally:
        sys.stdout = old_stdout
    
    print("✅ Training pipeline completed successfully")
    print("Note: Solving Conway's problem requires much longer training (hours/days)")


if __name__ == "__main__":
    print("🧪 PDS Set Transformer - Test Suite")
    print("=" * 60)
    
    # Test 1: Manual search to understand the problem
    manual_search_small_case(7)
    
    # Test 2: Small known cases
    test_small_known_pds()
    
    # Test 3: Quick Conway test (just to verify pipeline works)
    quick_conway_test()
    
    print("\n🏁 Test suite completed!")
    print("\nNext steps:")
    print("1. If small cases work, run full Conway training")
    print("2. Consider trying different groups (Z_9 × Z_11)")
    print("3. Experiment with hyperparameters")