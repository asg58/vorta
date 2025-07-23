# VORTA Factory Pattern Integration Test (Simple Version)

"""
Simplified Integration Test for VORTA Factory Pattern - Phase 5.5
No emoji characters to avoid encoding issues on Windows terminals.
"""

import os
import sys
import time
from pathlib import Path

def main():
    """Run simplified integration tests"""
    print("VORTA Factory Pattern Integration Test Suite")
    print("Phase 5.5: Integration Testing & Validation")
    print("=" * 60)
    
    # Setup
    workspace_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(workspace_path))
    os.environ["VORTA_ENVIRONMENT"] = "testing"
    
    # Test categories
    test_results = []
    
    # Test 1: Basic Factory Import
    print("\n1. Testing Factory Manager Import...")
    try:
        from frontend.components.factory_manager import get_factory_manager
        factory = get_factory_manager()
        print("   PASS: Factory Manager imported successfully")
        test_results.append(True)
    except Exception as e:
        print(f"   FAIL: Factory Manager import failed: {e}")
        test_results.append(False)
        return False
    
    # Test 2: Component Creation
    print("\n2. Testing Component Creation...")
    test_components = [
        "neural_vad_processor",
        "wake_word_detector", 
        "conversation_orchestrator",
        "real_time_audio_streamer"
    ]
    
    component_results = []
    for component in test_components:
        try:
            if hasattr(factory, f"create_{component}"):
                start_time = time.perf_counter()
                instance = getattr(factory, f"create_{component}")()
                end_time = time.perf_counter()
                
                creation_time = (end_time - start_time) * 1000
                
                if instance is not None:
                    print(f"   PASS: {component} created in {creation_time:.3f}ms")
                    component_results.append(True)
                    del instance
                else:
                    print(f"   FAIL: {component} returned None")
                    component_results.append(False)
            else:
                print(f"   FAIL: {component} factory method not found")
                component_results.append(False)
                
        except Exception as e:
            print(f"   FAIL: {component} creation failed: {e}")
            component_results.append(False)
    
    component_success = all(component_results)
    test_results.append(component_success)
    
    # Test 3: Environment Switching
    print("\n3. Testing Environment Switching...")
    try:
        environments = ["testing", "production", "testing"]
        switch_results = []
        
        for env in environments:
            os.environ["VORTA_ENVIRONMENT"] = env
            factory = get_factory_manager()
            
            # Test component creation in new environment
            component = factory.create_neural_vad_processor()
            if component is not None:
                print(f"   PASS: Environment {env} switch successful")
                switch_results.append(True)
                del component
            else:
                print(f"   FAIL: Environment {env} switch failed")
                switch_results.append(False)
        
        env_success = all(switch_results)
        test_results.append(env_success)
        
    except Exception as e:
        print(f"   FAIL: Environment switching failed: {e}")
        test_results.append(False)
    
    # Results Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS - Phase 5.5")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\nALL TESTS PASSED - Phase 5.5 Integration Testing SUCCESS")
        print("Factory Pattern implementation meets integration requirements")
        return True
    else:
        print("\nSOME TESTS FAILED - Phase 5.5 needs attention")
        print("Review failed components and address issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
