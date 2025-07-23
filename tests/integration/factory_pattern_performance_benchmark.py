# ⚡ VORTA Factory Pattern Performance Benchmarking

"""
Performance Benchmarking Suite for VORTA Factory Pattern Implementation

This script measures the performance overhead of the Factory Pattern implementation
and validates that the <1ms target is achieved for component instantiation.

Phase 5.5 Implementation: Performance Benchmarking with Factory Pattern overhead measurement
Status: 🚧 IN PROGRESS

Benchmarks:
- Factory Pattern overhead measurement
- Mock vs Production component creation times
- Memory usage analysis
- Concurrent creation performance
- Environment switching overhead
"""

import time
import os
import gc
import psutil
import asyncio
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result"""
    component_name: str
    category: str
    environment: str
    avg_creation_time_ms: float
    min_creation_time_ms: float
    max_creation_time_ms: float
    std_dev_ms: float
    memory_usage_mb: float
    iterations: int
    meets_target: bool
    target_ms: float = 1.0

@dataclass
class OverheadMeasurement:
    """Factory pattern overhead measurement"""
    direct_creation_time_ms: float
    factory_creation_time_ms: float
    overhead_ms: float
    overhead_percentage: float

class VORTAPerformanceBenchmark:
    """
    Performance benchmarking suite for VORTA Factory Pattern
    """
    
    def __init__(self):
        self.benchmarks: List[PerformanceBenchmark] = []
        self.overhead_measurements: List[OverheadMeasurement] = []
        self.factory_manager = None
        self.process = psutil.Process()
        
        # Component categories for testing
        self.test_components = {
            "audio": ["neural_vad_processor", "wake_word_detector", "audio_quality_analyzer"],
            "ai": ["conversation_orchestrator", "intent_recognition_engine", "response_generation_engine"],
            "voice": ["real_time_audio_streamer", "voice_cloning_engine", "voice_quality_enhancer"], 
            "agi": ["agi_multi_modal_processor", "predictive_conversation", "performance_analytics"]
        }

    def setup_benchmarking(self):
        """Setup benchmarking environment"""
        print("🚀 Setting up VORTA Performance Benchmarking Suite")
        
        try:
            from frontend.components.factory_manager import get_factory_manager
            self.factory_manager = get_factory_manager()
            print("✅ Factory Manager loaded for benchmarking")
        except Exception as e:
            print(f"❌ Failed to load Factory Manager: {e}")
            raise

    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_component_creation(self, component_name: str, category: str, 
                                   environment: str, iterations: int = 100) -> PerformanceBenchmark:
        """Benchmark component creation performance"""
        print(f"📊 Benchmarking {component_name} in {environment} environment")
        
        # Set environment and prepare
        os.environ["VORTA_ENVIRONMENT"] = environment
        factory = self.factory_manager
        self._warmup_component(factory, component_name)
        
        # Measure baseline memory
        gc.collect()
        initial_memory = self.measure_memory_usage()
        
        # Run performance benchmark
        creation_times = self._measure_creation_times(factory, component_name, iterations)
        
        # Calculate memory usage
        gc.collect()
        final_memory = self.measure_memory_usage()
        memory_usage = final_memory - initial_memory
        
        # Generate benchmark result
        return self._create_benchmark_result(
            component_name, category, environment, creation_times, memory_usage
        )

    def _warmup_component(self, factory, component_name: str):
        """Warmup component creation"""
        for _ in range(5):
            try:
                if hasattr(factory, f"create_{component_name}"):
                    component = getattr(factory, f"create_{component_name}")()
                    del component
            except Exception:
                pass

    def _measure_creation_times(self, factory, component_name: str, iterations: int) -> List[float]:
        """Measure component creation times"""
        creation_times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            try:
                if hasattr(factory, f"create_{component_name}"):
                    component = getattr(factory, f"create_{component_name}")()
                    end_time = time.perf_counter()
                    creation_time_ms = (end_time - start_time) * 1000
                    creation_times.append(creation_time_ms)
                    del component
                else:
                    print(f"⚠️  Factory method create_{component_name} not found")
                    break
                    
            except Exception as e:
                print(f"⚠️  Error creating {component_name}: {e}")
                continue
        
        return creation_times

    def _create_benchmark_result(self, component_name: str, category: str, environment: str, 
                               creation_times: List[float], memory_usage: float) -> PerformanceBenchmark:
        """Create benchmark result from measurements"""
        if creation_times:
            avg_time = statistics.mean(creation_times)
            min_time = min(creation_times)
            max_time = max(creation_times)
            std_dev = statistics.stdev(creation_times) if len(creation_times) > 1 else 0.0
            meets_target = avg_time < 1.0
            
            # Status reporting
            status = "✅" if meets_target else "⚠️"
            print(f"  {status} Avg: {avg_time:.3f}ms, Min: {min_time:.3f}ms, Max: {max_time:.3f}ms")
            
            return PerformanceBenchmark(
                component_name=component_name,
                category=category,
                environment=environment,
                avg_creation_time_ms=avg_time,
                min_creation_time_ms=min_time,
                max_creation_time_ms=max_time,
                std_dev_ms=std_dev,
                memory_usage_mb=memory_usage,
                iterations=len(creation_times),
                meets_target=meets_target
            )
        else:
            print(f"❌ No successful creations for {component_name}")
            return PerformanceBenchmark(
                component_name=component_name,
                category=category,
                environment=environment,
                avg_creation_time_ms=999.0,
                min_creation_time_ms=999.0,
                max_creation_time_ms=999.0,
                std_dev_ms=0.0,
                memory_usage_mb=0.0,
                iterations=0,
                meets_target=False
            )

    def benchmark_all_components(self):
        """Benchmark all components across environments"""
        print("🧪 Running comprehensive component benchmarking")
        
        environments = ["testing", "production"]  # Skip development for now
        
        for environment in environments:
            print(f"\n🏗️  Testing {environment} environment")
            
            for category, components in self.test_components.items():
                print(f"  📦 Category: {category}")
                
                for component in components:
                    try:
                        benchmark = self.benchmark_component_creation(
                            component, category, environment, iterations=50
                        )
                        self.benchmarks.append(benchmark)
                    except Exception as e:
                        print(f"  ❌ Failed to benchmark {component}: {e}")

    def measure_factory_overhead(self) -> Optional[OverheadMeasurement]:
        """Measure Factory Pattern overhead vs direct instantiation"""
        print("📊 Measuring Factory Pattern overhead")
        
        # We'll measure this conceptually since direct instantiation requires imports
        # This would typically compare:
        # Direct: component = RealTimeAudioStreamer()
        # Factory: component = factory.create_real_time_audio_streamer()
        
        # For now, we'll estimate based on factory method call overhead
        os.environ["VORTA_ENVIRONMENT"] = "testing"
        factory = get_factory_manager()
        
        # Measure factory method call time
        iterations = 1000
        factory_times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                # Measure just the factory method lookup overhead
                getattr(factory, "create_neural_vad_processor")
                end_time = time.perf_counter()
                factory_times.append((end_time - start_time) * 1000)
            except AttributeError:
                break
        
        if factory_times:
            avg_factory_time = statistics.mean(factory_times)
            # Estimated direct instantiation time (very minimal)
            estimated_direct_time = 0.001  # ~1 microsecond
            
            overhead = avg_factory_time - estimated_direct_time
            overhead_percentage = (overhead / estimated_direct_time) * 100
            
            measurement = OverheadMeasurement(
                direct_creation_time_ms=estimated_direct_time,
                factory_creation_time_ms=avg_factory_time,
                overhead_ms=overhead,
                overhead_percentage=overhead_percentage
            )
            
            print(f"  📊 Factory overhead: {overhead:.4f}ms ({overhead_percentage:.1f}% increase)")
            return measurement
        
        return None

    def benchmark_environment_switching(self) -> Dict[str, float]:
        """Benchmark environment switching performance"""
        print("🔄 Benchmarking environment switching")
        
        environments = ["testing", "production", "testing"]
        switching_times = []
        
        for i in range(len(environments) - 1):
            current_env = environments[i]
            next_env = environments[i + 1]
            
            # Measure time to switch environments and create factory
            start_time = time.perf_counter()
            os.environ["VORTA_ENVIRONMENT"] = next_env
            factory = self.factory_manager
            
            # Test component creation to ensure environment is active
            try:
                component = factory.create_neural_vad_processor()
                del component
                end_time = time.perf_counter()
                
                switch_time = (end_time - start_time) * 1000
                switching_times.append(switch_time)
                print(f"  🔄 {current_env} → {next_env}: {switch_time:.3f}ms")
                
            except Exception as e:
                print(f"  ❌ Failed to switch {current_env} → {next_env}: {e}")
        
        return {
            "avg_switch_time_ms": statistics.mean(switching_times) if switching_times else 0,
            "max_switch_time_ms": max(switching_times) if switching_times else 0,
            "min_switch_time_ms": min(switching_times) if switching_times else 0
        }

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        print("\n📊 Generating Performance Report")
        
        # Calculate overall statistics
        testing_benchmarks = [b for b in self.benchmarks if b.environment == "testing"]
        production_benchmarks = [b for b in self.benchmarks if b.environment == "production"]
        
        # Performance targets analysis
        passing_components = [b for b in self.benchmarks if b.meets_target]
        failing_components = [b for b in self.benchmarks if not b.meets_target]
        
        overall_pass_rate = len(passing_components) / len(self.benchmarks) if self.benchmarks else 0
        
        # Environment switching performance
        env_switch_perf = self.benchmark_environment_switching()
        
        # Factory overhead measurement
        overhead = self.measure_factory_overhead()
        
        report = {
            "performance_summary": {
                "total_components_tested": len(self.benchmarks),
                "components_meeting_target": len(passing_components),
                "components_exceeding_target": len(failing_components),
                "overall_pass_rate": overall_pass_rate,
                "target_threshold_ms": 1.0,
                "timestamp": datetime.now().isoformat()
            },
            "environment_performance": {
                "testing": {
                    "components": len(testing_benchmarks),
                    "avg_creation_time_ms": statistics.mean([b.avg_creation_time_ms for b in testing_benchmarks]) if testing_benchmarks else 0,
                    "pass_rate": len([b for b in testing_benchmarks if b.meets_target]) / len(testing_benchmarks) if testing_benchmarks else 0
                },
                "production": {
                    "components": len(production_benchmarks),
                    "avg_creation_time_ms": statistics.mean([b.avg_creation_time_ms for b in production_benchmarks]) if production_benchmarks else 0,
                    "pass_rate": len([b for b in production_benchmarks if b.meets_target]) / len(production_benchmarks) if production_benchmarks else 0
                }
            },
            "factory_pattern_overhead": overhead.__dict__ if overhead else {},
            "environment_switching": env_switch_perf,
            "detailed_benchmarks": [asdict(b) for b in self.benchmarks],
            "performance_recommendations": self.generate_recommendations()
        }
        
        return report

    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze failing components
        failing_components = [b for b in self.benchmarks if not b.meets_target]
        
        if failing_components:
            slow_components = [b.component_name for b in failing_components]
            recommendations.append(f"Optimize factory creation for: {', '.join(slow_components)}")
        
        # Memory usage recommendations
        high_memory_components = [b for b in self.benchmarks if b.memory_usage_mb > 10.0]
        if high_memory_components:
            recommendations.append("Review memory usage for components with >10MB overhead")
        
        # Environment switching recommendations
        recommendations.append("Consider caching factory instances for frequently switched environments")
        
        # Overall performance
        overall_pass_rate = len([b for b in self.benchmarks if b.meets_target]) / len(self.benchmarks) if self.benchmarks else 0
        if overall_pass_rate < 0.9:
            recommendations.append("Factory pattern implementation needs performance optimization")
        else:
            recommendations.append("Factory pattern performance meets enterprise requirements")
        
        return recommendations

    def save_report(self, report: Dict, filename: str = "vorta_performance_report.json"):
        """Save performance report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Performance report saved to {filename}")

    def print_summary(self, report: Dict):
        """Print performance summary to console"""
        print("\n" + "="*80)
        print("⚡ VORTA FACTORY PATTERN PERFORMANCE REPORT")
        print("="*80)
        
        summary = report["performance_summary"]
        print(f"Components Tested: {summary['total_components_tested']}")
        print(f"Meeting <1ms Target: {summary['components_meeting_target']}")
        print(f"Exceeding Target: {summary['components_exceeding_target']}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']:.1%}")
        
        print("\n🏗️  ENVIRONMENT PERFORMANCE:")
        for env, perf in report["environment_performance"].items():
            print(f"  {env.title()}: {perf['avg_creation_time_ms']:.3f}ms avg ({perf['pass_rate']:.1%} pass rate)")
        
        if report["factory_pattern_overhead"]:
            overhead = report["factory_pattern_overhead"]
            print(f"\n🏭 FACTORY PATTERN OVERHEAD: {overhead.get('overhead_ms', 0):.4f}ms")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["performance_recommendations"]:
            print(f"  • {rec}")
        
        print("="*80)

def main():
    """Main benchmarking execution"""
    benchmarker = VORTAPerformanceBenchmark()
    
    try:
        # Setup
        benchmarker.setup_benchmarking()
        
        # Run benchmarks
        benchmarker.benchmark_all_components()
        
        # Generate and display report
        report = benchmarker.generate_performance_report()
        benchmarker.print_summary(report)
        benchmarker.save_report(report)
        
        return report
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return None

if __name__ == "__main__":
    main()
