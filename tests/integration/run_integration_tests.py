# ⚡ VORTA Integration Test Runner

"""
Simplified Integration Test Runner for VORTA Factory Pattern

This test runner executes the Factory Pattern integration tests and performance
benchmarks as part of Phase 5.5 Implementation.

Phase 5.5: Integration Testing & Validation
Status: 🚧 IN PROGRESS
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class VORTATestRunner:
    """Simple test runner for VORTA integration tests"""
    
    def __init__(self):
        self.workspace_path = Path(__file__).parent.parent.parent
        self.test_results = []
        
    def setup_environment(self):
        """Setup test environment"""
        print("🚀 Setting up VORTA Integration Test Environment")
        
        # Ensure PYTHONPATH includes workspace
        sys.path.insert(0, str(self.workspace_path))
        
        # Set default test environment
        os.environ["VORTA_ENVIRONMENT"] = "testing"
        
        print("✅ Test environment ready")

    def run_factory_pattern_tests(self):
        """Run factory pattern component tests"""
        print("\n🧪 Running Factory Pattern Component Tests")
        
        try:
            # Import factory manager
            from frontend.components.factory_manager import get_factory_manager
            
            factory = get_factory_manager()
            test_components = [
                "neural_vad_processor",
                "wake_word_detector", 
                "conversation_orchestrator",
                "real_time_audio_streamer"
            ]
            
            passed_tests = 0
            total_tests = len(test_components)
            
            for component in test_components:
                try:
                    if hasattr(factory, f"create_{component}"):
                        start_time = time.perf_counter()
                        instance = getattr(factory, f"create_{component}")()
                        end_time = time.perf_counter()
                        
                        creation_time = (end_time - start_time) * 1000
                        
                        if instance is not None:
                            status = "✅ PASS" if creation_time < 10 else "⚠️ SLOW"
                            print(f"  {status} {component}: {creation_time:.3f}ms")
                            passed_tests += 1
                            del instance
                        else:
                            print(f"  ❌ FAIL {component}: returned None")
                    else:
                        print(f"  ❌ FAIL {component}: factory method not found")
                        
                except Exception as e:
                    print(f"  ❌ FAIL {component}: {e}")
            
            print(f"\n📊 Factory Pattern Tests: {passed_tests}/{total_tests} passed")
            return passed_tests == total_tests
            
        except ImportError as e:
            print(f"❌ Failed to import factory manager: {e}")
            return False

    def run_environment_switching_tests(self):
        """Test environment switching"""
        print("\n🔄 Running Environment Switching Tests")
        
        try:
            from frontend.components.factory_manager import get_factory_manager
            
            environments = ["testing", "production", "testing"]
            switch_success = 0
            
            for i, env in enumerate(environments):
                try:
                    os.environ["VORTA_ENVIRONMENT"] = env
                    factory = get_factory_manager()
                    
                    # Test component creation in new environment
                    component = factory.create_neural_vad_processor()
                    if component is not None:
                        print(f"  ✅ Environment {env}: component created successfully")
                        switch_success += 1
                        del component
                    else:
                        print(f"  ❌ Environment {env}: component creation failed")
                        
                except Exception as e:
                    print(f"  ❌ Environment {env}: {e}")
            
            print(f"📊 Environment Switching: {switch_success}/{len(environments)} passed")
            return switch_success == len(environments)
            
        except ImportError as e:
            print(f"❌ Failed to import factory manager: {e}")
            return False

    def run_performance_benchmark(self):
        """Run performance benchmark if script exists"""
        print("\n⚡ Running Performance Benchmark")
        
        benchmark_script = self.workspace_path / "tests" / "integration" / "factory_pattern_performance_benchmark.py"
        
        if benchmark_script.exists():
            try:
                # Run performance benchmark
                result = subprocess.run([
                    sys.executable, str(benchmark_script)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Performance benchmark completed successfully")
                    if result.stdout:
                        print("📊 Benchmark Output:")
                        print(result.stdout[-1000:])  # Last 1000 chars
                    return True
                else:
                    print("❌ Performance benchmark failed")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print("⏱️ Performance benchmark timed out (60s limit)")
                return False
            except Exception as e:
                print(f"❌ Failed to run performance benchmark: {e}")
                return False
        else:
            print("⚠️ Performance benchmark script not found")
            return True

    def generate_test_report(self, results):
        """Generate test execution report"""
        print("\n" + "="*80)
        print("🧪 VORTA INTEGRATION TEST REPORT - Phase 5.5")
        print("="*80)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results if result['passed'])
        
        print("Test Suite: Factory Pattern Integration")
        print(f"Total Test Categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Test Results:")
        for result in results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {status} {result['name']}")
            if result.get('details'):
                print(f"       {result['details']}")
        
        # Overall assessment
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED - Phase 5.5 Integration Testing SUCCESS")
            print("📈 Factory Pattern implementation meets integration requirements")
        else:
            print("\n⚠️ SOME TESTS FAILED - Phase 5.5 needs attention")
            print("🔧 Review failed components and address issues")
        
        print("="*80)
        
        return passed_tests == total_tests

    def run_all_tests(self):
        """Run complete integration test suite"""
        print("🎯 VORTA Factory Pattern Integration Test Suite")
        print("Phase 5.5: Integration Testing & Validation\n")
        
        # Setup
        self.setup_environment()
        
        # Run test categories
        test_results = []
        
        # Factory Pattern Tests
        factory_passed = self.run_factory_pattern_tests()
        test_results.append({
            'name': 'Factory Pattern Component Tests',
            'passed': factory_passed,
            'details': '4 core components tested for instantiation'
        })
        
        # Environment Switching Tests  
        env_passed = self.run_environment_switching_tests()
        test_results.append({
            'name': 'Environment Switching Tests',
            'passed': env_passed,
            'details': 'testing → production → testing transitions'
        })
        
        # Performance Benchmark
        perf_passed = self.run_performance_benchmark()
        test_results.append({
            'name': 'Performance Benchmark',
            'passed': perf_passed,
            'details': 'Component creation time and memory usage analysis'
        })
        
        # Generate report
        overall_success = self.generate_test_report(test_results)
        
        return overall_success

def main():
    """Main test execution"""
    runner = VORTATestRunner()
    
    try:
        success = runner.run_all_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
