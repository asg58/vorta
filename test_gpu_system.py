#!/usr/bin/env python3
"""Test Multi-Tenant Architecture with 8GB GPU."""

import asyncio
import sys
import traceback
from services.multi_tenancy.multi_tenant_architecture import (
    MultiTenantArchitecture, WorkloadType, ProcessingUnit, ProcessingTask
)

async def test_gpu_system():
    """Test the multi-tenant system with real GPU acceleration."""
    print('🚀 Testing Multi-Tenant System with RTX 4060 GPU')
    print('=' * 60)
    
    try:
        arch = MultiTenantArchitecture()
        
        # Display GPU resource summary
        summary = arch.resource_manager.get_resource_summary()
        print(f'🎮 GPU Available: {summary["gpu"]["available"]}')
        print(f'🎮 GPU Name: RTX 4060 Laptop GPU')
        print(f'🎮 GPU Memory: {summary["gpu"]["memory_total_gb"]:.1f}GB')
        print(f'🎮 CUDA Available: {summary["gpu"]["cuda_available"]}')
        print(f'💻 CPU Cores: {summary["cpu"]["cores"]}')
        print(f'💾 System Memory: {summary["memory"]["total_gb"]:.1f}GB')
        print()
        
        # Test tasks for different processing units
        test_tasks = [
            ('GPU Inference Task', WorkloadType.INFERENCE, ProcessingUnit.GPU),
            ('CPU Database Task', WorkloadType.DATABASE, ProcessingUnit.CPU),
            ('Hybrid Audio Task', WorkloadType.AUDIO_PROCESSING, ProcessingUnit.HYBRID),
            ('GPU Analytics Task', WorkloadType.ANALYTICS, ProcessingUnit.GPU),
            ('Hybrid Streaming Task', WorkloadType.STREAMING, ProcessingUnit.HYBRID),
            ('GPU Text Processing', WorkloadType.TEXT_PROCESSING, ProcessingUnit.GPU)
        ]
        
        print('🎯 Running 8GB GPU Optimized Tasks:')
        print('-' * 60)
        
        results = []
        total_gpu_time = 0.0
        total_cpu_time = 0.0
        total_hybrid_time = 0.0
        
        for i, (name, workload, processing) in enumerate(test_tasks):
            try:
                # Create task object
                task = ProcessingTask(
                    task_id=f'task_{i}',
                    tenant_id='test_tenant',
                    workload_type=workload,
                    data={},
                    processing_unit=processing
                )
                
                # Process task based on type
                if processing == ProcessingUnit.GPU:
                    result = await arch.resource_manager.process_task_gpu(task)
                    total_gpu_time += result["processing_time"]
                elif processing == ProcessingUnit.HYBRID:
                    result = await arch.resource_manager.process_task_hybrid(task)
                    total_hybrid_time += result["processing_time"]
                else:
                    result = await arch.resource_manager.process_task_cpu(task)
                    total_cpu_time += result["processing_time"]
                
                results.append(result)
                
                # Display result
                status = "✅" if result["status"] == "completed" else "❌"
                print(f'{status} {name}: {result["processing_unit"]} - {result["processing_time"]:.3f}s')
                
                # Show GPU memory usage if available
                if isinstance(result.get('result'), dict):
                    gpu_mem = result["result"].get("gpu_memory_used")
                    if gpu_mem:
                        print(f'   💾 GPU Memory Used: {gpu_mem}')
                    
                    speedup = result["result"].get("speedup")
                    if speedup:
                        print(f'   ⚡ Speedup: {speedup}')
                
            except Exception as e:
                print(f'❌ {name}: Error - {e}')
                traceback.print_exc()
        
        print()
        print('📊 Performance Summary:')
        print('-' * 40)
        
        gpu_tasks = len([r for r in results if r["processing_unit"] == "GPU"])
        cpu_tasks = len([r for r in results if r["processing_unit"] == "CPU"])
        hybrid_tasks = len([r for r in results if r["processing_unit"] == "HYBRID"])
        
        print(f'Total Tasks: {len(results)}')
        print(f'  GPU Tasks: {gpu_tasks} - Total Time: {total_gpu_time:.3f}s')
        print(f'  CPU Tasks: {cpu_tasks} - Total Time: {total_cpu_time:.3f}s') 
        print(f'  Hybrid Tasks: {hybrid_tasks} - Total Time: {total_hybrid_time:.3f}s')
        
        if gpu_tasks > 0:
            print(f'  Avg GPU Time: {total_gpu_time/gpu_tasks:.3f}s')
        if cpu_tasks > 0:
            print(f'  Avg CPU Time: {total_cpu_time/cpu_tasks:.3f}s')
        if hybrid_tasks > 0:
            print(f'  Avg Hybrid Time: {total_hybrid_time/hybrid_tasks:.3f}s')
        
        print()
        print('📊 GPU Memory Summary:')
        print('-' * 30)
        gpu_stats = arch.resource_manager.get_gpu_memory_usage()
        print(f'Total GPU Memory: {gpu_stats["total_gb"]:.1f}GB')
        print(f'Allocated: {gpu_stats["allocated_gb"]:.1f}GB')
        print(f'Available: {gpu_stats["available_gb"]:.1f}GB') 
        print(f'Usage: {gpu_stats["usage_percent"]:.1f}%')
        print(f'Reserved for System: {gpu_stats["reserved_gb"]:.1f}GB')
        
    except Exception as e:
        print(f'❌ System Error: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gpu_system())
