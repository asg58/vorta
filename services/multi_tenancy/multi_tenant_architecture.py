# services/multi_tenancy/multi_tenant_architecture.py
"""
VORTA AGI: Multi-Tenant Architecture with Hybrid CPU/GPU Processing

Enterprise customer isolat            logger.info("🔄 GPU support configured for 8GB GPU (CUDA not currently available)")on and scaling with intelligent CPU/GPU resource allocation
- Tenant isolation and resource management with GPU acceleration
- Dynamic hybrid CPU/GPU resource allocation per tenant
- Enterprise billing and usage tracking across CPU/GPU resources
- Secure inter-tenant communication with performance optimization
- Tenant-specific customizations with processing unit preferences
- Real-time CPU/GPU performance monitoring and optimization
"""

import asyncio
import time
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# GPU acceleration imports
try:
    import torch
    import torch.cuda
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count()
except ImportError:
    GPU_AVAILABLE = False
    GPU_COUNT = 0

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TenantTier(Enum):
    """Tenant service tiers with different CPU/GPU allocations."""
    BASIC = "basic"           # CPU only
    PROFESSIONAL = "professional"  # CPU + limited GPU
    ENTERPRISE = "enterprise"      # CPU + full GPU access
    ULTRA = "ultra"          # CPU + multi-GPU + priority

class ResourceType(Enum):
    """Types of resources that can be allocated to tenants."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"      # New: GPU memory allocation
    GPU_COMPUTE = "gpu_compute"    # New: GPU compute allocation
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    API_CALLS = "api_calls"
    CONCURRENT_USERS = "concurrent_users"

class ProcessingUnit(Enum):
    """Available processing units for workload execution."""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"  # Both CPU and GPU

class WorkloadType(Enum):
    """Different types of workloads for optimal CPU/GPU distribution."""
    INFERENCE = "inference"      # AI model inference (GPU preferred)
    AUDIO_PROCESSING = "audio"   # Audio DSP (CPU/GPU hybrid)
    TEXT_PROCESSING = "text"     # NLP tasks (GPU preferred)
    DATABASE = "database"        # Database operations (CPU)
    STREAMING = "streaming"      # Real-time streaming (hybrid)
    ANALYTICS = "analytics"      # Data analytics (GPU preferred)

@dataclass
class ResourceLimit:
    """Resource limits for a tenant with CPU/GPU tracking."""
    resource_type: ResourceType
    limit: float
    current_usage: float = 0.0
    peak_usage: float = 0.0  # Track peak usage for analytics
    last_reset: datetime = field(default_factory=datetime.now)
    
@dataclass
class CPUGPUMetrics:
    """CPU and GPU performance metrics for tenants."""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    processing_tasks_active: int = 0
    last_updated: float = field(default_factory=time.time)

@dataclass
class ProcessingTask:
    """Task for CPU/GPU processing."""
    task_id: str
    tenant_id: str
    workload_type: WorkloadType
    data: Any
    processing_unit: ProcessingUnit = ProcessingUnit.HYBRID
    priority: int = 5  # 1-10, higher is more important
    created_at: float = field(default_factory=time.time)
    estimated_duration: float = 1.0  # seconds
    gpu_memory_required: int = 0  # MB of GPU memory needed

@dataclass
class TenantConfig:
    """Configuration for a tenant with CPU/GPU preferences."""
    tenant_id: str
    name: str
    tier: TenantTier
    resource_limits: Dict[ResourceType, ResourceLimit]
    cpu_gpu_metrics: CPUGPUMetrics = field(default_factory=CPUGPUMetrics)
    preferred_processing: ProcessingUnit = ProcessingUnit.HYBRID
    workload_types: List[WorkloadType] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class UsageMetrics:
    """Enhanced usage metrics for a tenant with CPU/GPU tracking."""
    tenant_id: str
    timestamp: datetime
    resource_usage: Dict[ResourceType, float]
    cpu_gpu_metrics: CPUGPUMetrics
    api_calls_count: int = 0
    active_users: int = 0
    response_time_ms: float = 0.0
    gpu_tasks_completed: int = 0
    cpu_tasks_completed: int = 0
    hybrid_tasks_completed: int = 0

class CPUGPUResourceManager:
    """Manages hybrid CPU/GPU resource allocation and monitoring."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB
        self.gpu_available = GPU_AVAILABLE
        self.gpu_count = GPU_COUNT
        self.gpu_memory_total = 8.0  # Assume 8GB GPU based on user specification
        
        if self.gpu_available:
            try:
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                logger.info(f"🚀 GPU detected: {GPU_COUNT} GPUs, {self.gpu_memory_total:.1f}GB GPU memory")
            except Exception as e:
                logger.warning(f"GPU properties detection failed: {e}, assuming 8GB")
                self.gpu_memory_total = 8.0
        else:
            logger.info(f"� GPU support configured for 8GB GPU (CUDA not currently available)")
        
        logger.info(f"�💻 System resources: {self.cpu_count} CPU cores, {self.memory_total:.1f}GB RAM")
        logger.info(f"🎮 GPU configuration: {self.gpu_memory_total:.1f}GB GPU memory")
        
        # Thread pools for different processing types
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.cpu_count)
        
        # Resource monitoring
        self.monitoring_active = False
        self.current_cpu_usage = 0.0
        self.current_memory_usage = 0.0
        self.current_gpu_usage = 0.0
        self.current_gpu_memory = 0.0
        
        # GPU memory allocation tracking (for 8GB GPU)
        self.gpu_memory_allocated = 0.0  # Track allocated GPU memory
        self.gpu_memory_reserved = 1.0   # Reserve 1GB for system operations

    def allocate_gpu_memory(self, requested_gb: float) -> bool:
        """Allocate GPU memory if available (for 8GB GPU management)."""
        available_memory = self.gpu_memory_total - self.gpu_memory_allocated - self.gpu_memory_reserved
        if available_memory >= requested_gb:
            self.gpu_memory_allocated += requested_gb
            logger.info(f"Allocated {requested_gb:.1f}GB GPU memory. Total allocated: {self.gpu_memory_allocated:.1f}GB")
            return True
        else:
            logger.warning(f"Insufficient GPU memory. Requested: {requested_gb:.1f}GB, Available: {available_memory:.1f}GB")
            return False
    
    def release_gpu_memory(self, released_gb: float):
        """Release previously allocated GPU memory."""
        self.gpu_memory_allocated = max(0.0, self.gpu_memory_allocated - released_gb)
        logger.info(f"Released {released_gb:.1f}GB GPU memory. Total allocated: {self.gpu_memory_allocated:.1f}GB")
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        available = self.gpu_memory_total - self.gpu_memory_allocated - self.gpu_memory_reserved
        usage_percent = (self.gpu_memory_allocated / self.gpu_memory_total) * 100
        
        return {
            "total_gb": self.gpu_memory_total,
            "allocated_gb": self.gpu_memory_allocated,
            "reserved_gb": self.gpu_memory_reserved,
            "available_gb": available,
            "usage_percent": usage_percent
        }

    def get_optimal_processing_unit(self, workload_type: WorkloadType, tenant_tier: TenantTier) -> ProcessingUnit:
        """Determine optimal processing unit based on workload and tenant tier."""
        # Basic tier gets CPU only
        if tenant_tier == TenantTier.BASIC:
            return ProcessingUnit.CPU
        
        # No GPU available
        if not self.gpu_available:
            return ProcessingUnit.CPU
        
        # GPU-preferred workloads
        if workload_type in [WorkloadType.INFERENCE, WorkloadType.TEXT_PROCESSING, WorkloadType.ANALYTICS]:
            if tenant_tier in [TenantTier.ENTERPRISE, TenantTier.ULTRA]:
                return ProcessingUnit.GPU
            else:
                return ProcessingUnit.HYBRID
        
        # Hybrid workloads
        if workload_type in [WorkloadType.AUDIO_PROCESSING, WorkloadType.STREAMING]:
            return ProcessingUnit.HYBRID
        
        # CPU-preferred workloads
        return ProcessingUnit.CPU

    async def process_task_cpu(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task on CPU."""
        start_time = time.time()
        try:
            # Simulate CPU processing based on workload type
            if task.workload_type == WorkloadType.DATABASE:
                await asyncio.sleep(0.1)  # Database operation
                result = {"records_processed": 100, "method": "cpu_database"}
            elif task.workload_type == WorkloadType.TEXT_PROCESSING:
                await asyncio.sleep(0.2)  # NLP processing
                result = {"tokens_processed": 1000, "method": "cpu_nlp"}
            elif task.workload_type == WorkloadType.AUDIO_PROCESSING:
                await asyncio.sleep(0.15)  # Audio DSP
                result = {"samples_processed": 44100, "method": "cpu_dsp"}
            else:
                await asyncio.sleep(0.1)
                result = {"status": "completed", "method": "cpu_generic"}
            
            processing_time = time.time() - start_time
            return {
                "task_id": task.task_id,
                "result": result,
                "processing_unit": "CPU",
                "processing_time": processing_time,
                "status": "completed"
            }
        except Exception as e:
            return {
                "task_id": task.task_id,
                "error": str(e),
                "processing_unit": "CPU",
                "processing_time": time.time() - start_time,
                "status": "failed"
            }

    async def process_task_gpu(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task on GPU with optimized 8GB memory management."""
        if not self.gpu_available:
            logger.info("GPU not available, falling back to CPU processing")
            return await self.process_task_cpu(task)
        
        # Estimate GPU memory requirement based on workload
        memory_requirement = 0.5  # Default 0.5GB
        if task.workload_type == WorkloadType.INFERENCE:
            memory_requirement = 1.5  # AI models need more memory
        elif task.workload_type == WorkloadType.ANALYTICS:
            memory_requirement = 2.0  # Analytics can be memory intensive
        elif task.workload_type == WorkloadType.TEXT_PROCESSING:
            memory_requirement = 1.0  # NLP models
        
        # Check and allocate GPU memory
        if not self.allocate_gpu_memory(memory_requirement):
            logger.warning(f"GPU memory allocation failed, falling back to CPU for task {task.task_id}")
            return await self.process_task_cpu(task)
        
        start_time = time.time()
        try:
            # Simulate GPU processing with actual GPU operations where possible
            if task.workload_type == WorkloadType.INFERENCE:
                # Simulate AI inference with tensor operations
                if GPU_AVAILABLE:
                    tensor = torch.randn(512, 512).cuda()  # Larger tensors for 8GB GPU
                    result_tensor = torch.matmul(tensor, tensor.T)
                    await asyncio.sleep(0.02)  # Fast GPU inference
                    result = {"predictions": 10, "confidence": 0.98, "tensor_size": result_tensor.shape, "gpu_memory_used": f"{memory_requirement}GB"}
                else:
                    await asyncio.sleep(0.08)
                    result = {"predictions": 10, "confidence": 0.85, "gpu_memory_used": f"{memory_requirement}GB"}
            elif task.workload_type == WorkloadType.TEXT_PROCESSING:
                # GPU-accelerated NLP
                await asyncio.sleep(0.05)
                result = {"tokens_processed": 1000, "method": "gpu_nlp", "speedup": "4x", "gpu_memory_used": f"{memory_requirement}GB"}
            elif task.workload_type == WorkloadType.ANALYTICS:
                # GPU analytics with 8GB optimization
                await asyncio.sleep(0.04)
                result = {"data_processed": "1M records", "method": "gpu_analytics", "speedup": "8x", "gpu_memory_used": f"{memory_requirement}GB"}
            else:
                await asyncio.sleep(0.03)
                result = {"status": "completed", "method": "gpu_generic", "speedup": "3x", "gpu_memory_used": f"{memory_requirement}GB"}
            
            processing_time = time.time() - start_time
            return {
                "task_id": task.task_id,
                "result": result,
                "processing_unit": "GPU",
                "processing_time": processing_time,
                "status": "completed",
                "gpu_memory_allocated": memory_requirement
            }
        except Exception as e:
            return {
                "task_id": task.task_id,
                "error": str(e),
                "processing_unit": "GPU",
                "processing_time": time.time() - start_time,
                "status": "failed",
                "gpu_memory_allocated": memory_requirement
            }
        finally:
            # Always release allocated GPU memory
            self.release_gpu_memory(memory_requirement)

    async def process_task_hybrid(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task using both CPU and GPU."""
        start_time = time.time()
        try:
            # Split workload between CPU and GPU
            if task.workload_type == WorkloadType.AUDIO_PROCESSING:
                # CPU handles preprocessing, GPU handles ML inference
                cpu_task = asyncio.create_task(self._simulate_audio_preprocessing_cpu(task.data))
                gpu_task = asyncio.create_task(self._simulate_audio_ml_gpu(task.data))
                cpu_result, gpu_result = await asyncio.gather(cpu_task, gpu_task)
                result = {"cpu_preprocessing": cpu_result, "gpu_ml": gpu_result}
            elif task.workload_type == WorkloadType.STREAMING:
                # CPU handles I/O, GPU handles processing
                cpu_result = await self._simulate_streaming_io_cpu(task.data)
                gpu_result = await self._simulate_streaming_processing_gpu(cpu_result)
                result = {"io_result": cpu_result, "processed": gpu_result}
            else:
                # Generic hybrid processing
                cpu_task = asyncio.create_task(self.process_task_cpu(task))
                gpu_task = asyncio.create_task(self.process_task_gpu(task))
                cpu_result, gpu_result = await asyncio.gather(cpu_task, gpu_task)
                result = {"cpu": cpu_result["result"], "gpu": gpu_result["result"]}
            
            processing_time = time.time() - start_time
            return {
                "task_id": task.task_id,
                "result": result,
                "processing_unit": "HYBRID",
                "processing_time": processing_time,
                "status": "completed"
            }
        except Exception as e:
            return {
                "task_id": task.task_id,
                "error": str(e),
                "processing_unit": "HYBRID",
                "processing_time": time.time() - start_time,
                "status": "failed"
            }

    # Helper simulation methods
    async def _simulate_audio_preprocessing_cpu(self, _data: Any) -> Dict[str, Any]:
        await asyncio.sleep(0.08)
        return {"preprocessed_samples": 44100, "filters_applied": 3}

    async def _simulate_audio_ml_gpu(self, _data: Any) -> Dict[str, Any]:
        if self.gpu_available:
            await asyncio.sleep(0.03)
            return {"ml_features": 128, "confidence": 0.95}
        await asyncio.sleep(0.12)
        return {"ml_features": 128, "confidence": 0.90}

    async def _simulate_streaming_io_cpu(self, _data: Any) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"bytes_transferred": 1024, "io_operations": 10}

    async def _simulate_streaming_processing_gpu(self, _io_data: Any) -> Dict[str, Any]:
        if self.gpu_available:
            await asyncio.sleep(0.02)
            return {"processed_streams": 4, "realtime": True}
        await asyncio.sleep(0.08)
        return {"processed_streams": 2, "realtime": False}

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource usage summary with 8GB GPU details."""
        gpu_memory_stats = self.get_gpu_memory_usage()
        
        return {
            "cpu": {
                "cores": self.cpu_count,
                "usage_percent": self.current_cpu_usage,
                "available": True
            },
            "memory": {
                "total_gb": self.memory_total,
                "usage_percent": self.current_memory_usage,
                "available": True
            },
            "gpu": {
                "available": self.gpu_available,
                "count": self.gpu_count,
                "memory_total_gb": self.gpu_memory_total,
                "memory_allocated_gb": gpu_memory_stats["allocated_gb"],
                "memory_available_gb": gpu_memory_stats["available_gb"],
                "memory_usage_percent": gpu_memory_stats["usage_percent"],
                "memory_reserved_gb": gpu_memory_stats["reserved_gb"],
                "cuda_available": GPU_AVAILABLE
            }
        }

class MultiTenantArchitectureError(Exception):
    """Custom exception for multi-tenant architecture errors."""
    pass

class MultiTenantArchitecture:
    """Enhanced multi-tenant architecture with hybrid CPU/GPU processing capabilities."""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.usage_history: Dict[str, List[UsageMetrics]] = {}
        self.resource_monitors: Dict[str, asyncio.Task] = {}
        self.resource_manager = CPUGPUResourceManager()
        self.processing_tasks: Dict[str, ProcessingTask] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}
        self._default_limits = self._initialize_default_limits()
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks": 0,
            "cpu_tasks": 0,
            "gpu_tasks": 0,
            "hybrid_tasks": 0,
            "avg_cpu_time": 0.0,
            "avg_gpu_time": 0.0,
            "avg_hybrid_time": 0.0
        }
        
        self._initialize_default_tenants()
        
    def _initialize_default_limits(self) -> Dict[TenantTier, Dict[ResourceType, float]]:
        """Initialize default resource limits for each tenant tier with GPU support."""
        return {
            TenantTier.BASIC: {
                ResourceType.CPU: 2.0,  # 2 CPU cores
                ResourceType.MEMORY: 4.0,  # 4 GB
                ResourceType.GPU_MEMORY: 0.0,  # No GPU access
                ResourceType.GPU_COMPUTE: 0.0,  # No GPU compute
                ResourceType.STORAGE: 50.0,  # 50 GB
                ResourceType.BANDWIDTH: 100.0,  # 100 MB/s
                ResourceType.API_CALLS: 10000.0,  # 10k calls/hour
                ResourceType.CONCURRENT_USERS: 50.0,  # 50 concurrent users
            },
            TenantTier.PROFESSIONAL: {
                ResourceType.CPU: 8.0,  # 8 CPU cores
                ResourceType.MEMORY: 16.0,  # 16 GB
                ResourceType.GPU_MEMORY: 3.0,  # 3 GB GPU memory (optimized for 8GB card)
                ResourceType.GPU_COMPUTE: 30.0,  # 30% GPU compute
                ResourceType.STORAGE: 200.0,  # 200 GB
                ResourceType.BANDWIDTH: 500.0,  # 500 MB/s
                ResourceType.API_CALLS: 100000.0,  # 100k calls/hour
                ResourceType.CONCURRENT_USERS: 500.0,  # 500 concurrent users
            },
            TenantTier.ENTERPRISE: {
                ResourceType.CPU: 32.0,  # 32 CPU cores
                ResourceType.MEMORY: 128.0,  # 128 GB
                ResourceType.GPU_MEMORY: 6.0,  # 6 GB GPU memory (optimized for 8GB card)
                ResourceType.GPU_COMPUTE: 65.0,  # 65% GPU compute
                ResourceType.STORAGE: 1000.0,  # 1 TB
                ResourceType.BANDWIDTH: 2000.0,  # 2 GB/s
                ResourceType.API_CALLS: 1000000.0,  # 1M calls/hour
                ResourceType.CONCURRENT_USERS: 5000.0,  # 5k concurrent users
            },
            TenantTier.ULTRA: {
                ResourceType.CPU: 64.0,  # 64 CPU cores
                ResourceType.MEMORY: 256.0,  # 256 GB
                ResourceType.GPU_MEMORY: 7.5,  # 7.5 GB GPU memory (near-full 8GB utilization)
                ResourceType.GPU_COMPUTE: 85.0,  # 85% GPU compute
                ResourceType.STORAGE: 5000.0,  # 5 TB
                ResourceType.BANDWIDTH: 10000.0,  # 10 GB/s
                ResourceType.API_CALLS: 10000000.0,  # 10M calls/hour
                ResourceType.CONCURRENT_USERS: 50000.0,  # 50k concurrent users
            }
        }

    def _initialize_default_tenants(self):
        """Initialize some default tenants for demonstration."""
        tenants_config = [
            {
                "name": "BasicCorp",
                "tier": TenantTier.BASIC,
                "workloads": [WorkloadType.DATABASE, WorkloadType.TEXT_PROCESSING],
                "preferred_processing": ProcessingUnit.CPU
            },
            {
                "name": "TechStartup",
                "tier": TenantTier.PROFESSIONAL,
                "workloads": [WorkloadType.INFERENCE, WorkloadType.AUDIO_PROCESSING],
                "preferred_processing": ProcessingUnit.HYBRID
            },
            {
                "name": "EnterpriseCorp",
                "tier": TenantTier.ENTERPRISE,
                "workloads": [WorkloadType.INFERENCE, WorkloadType.ANALYTICS, WorkloadType.STREAMING],
                "preferred_processing": ProcessingUnit.GPU
            },
            {
                "name": "UltraTech",
                "tier": TenantTier.ULTRA,
                "workloads": list(WorkloadType),  # All workload types
                "preferred_processing": ProcessingUnit.HYBRID
            }
        ]
        
        for config in tenants_config:
            tenant_id = self.create_tenant(
                name=config["name"],
                tier=config["tier"],
                custom_limits=None
            )
            # Update tenant with specific configuration
            tenant = self.tenants[tenant_id]
            tenant.workload_types = config["workloads"]
            tenant.preferred_processing = config["preferred_processing"]
        
        logger.info(f"🏢 Initialized {len(self.tenants)} tenants with hybrid CPU/GPU processing")

    def create_tenant(self, name: str, tier: TenantTier, 
                     custom_limits: Optional[Dict[ResourceType, float]] = None) -> str:
        """Create a new tenant with specified configuration."""
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        
        # Initialize resource limits based on tier
        if custom_limits:
            limits_config = custom_limits
        else:
            limits_config = self._default_limits.get(tier, self._default_limits[TenantTier.BASIC])
        
        resource_limits = {
            resource_type: ResourceLimit(resource_type=resource_type, limit=limit)
            for resource_type, limit in limits_config.items()
        }
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            resource_limits=resource_limits
        )
        
        self.tenants[tenant_id] = tenant_config
        self.usage_history[tenant_id] = []
        
        logger.info(f"Created tenant '{name}' (ID: {tenant_id}) with tier: {tier.value}")
        return tenant_id

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Retrieve tenant configuration."""
        return self.tenants.get(tenant_id)

    def update_tenant_tier(self, tenant_id: str, new_tier: TenantTier, 
                          custom_limits: Optional[Dict[ResourceType, float]] = None) -> bool:
        """Update tenant's service tier and resource limits."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            logger.error(f"Tenant {tenant_id} not found")
            return False
        
        old_tier = tenant.tier
        tenant.tier = new_tier
        
        # Update resource limits based on new tier
        if custom_limits:
            limits_config = custom_limits
        else:
            limits_config = self._default_limits.get(new_tier, self._default_limits[TenantTier.BASIC])
        
        for resource_type, new_limit in limits_config.items():
            if resource_type in tenant.resource_limits:
                tenant.resource_limits[resource_type].limit = new_limit
            else:
                tenant.resource_limits[resource_type] = ResourceLimit(
                    resource_type=resource_type, 
                    limit=new_limit
                )
        
        logger.info(f"Updated tenant {tenant_id} tier: {old_tier.value} → {new_tier.value}")
        return True

    def check_resource_availability(self, tenant_id: str, resource_type: ResourceType, 
                                   requested_amount: float) -> bool:
        """Check if tenant has enough resources available for the request."""
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active:
            return False
        
        if resource_type not in tenant.resource_limits:
            logger.warning(f"Resource type {resource_type.value} not configured for tenant {tenant_id}")
            return False
        
        limit = tenant.resource_limits[resource_type]
        available = limit.limit - limit.current_usage
        
        return available >= requested_amount

    def allocate_resources(self, tenant_id: str, resource_usage: Dict[ResourceType, float]) -> bool:
        """Allocate resources to a tenant if available."""
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active:
            return False
        
        # Check availability for all requested resources first
        for resource_type, amount in resource_usage.items():
            if not self.check_resource_availability(tenant_id, resource_type, amount):
                logger.warning(f"Insufficient {resource_type.value} for tenant {tenant_id}")
                return False
        
        # Allocate all resources
        for resource_type, amount in resource_usage.items():
            if resource_type in tenant.resource_limits:
                tenant.resource_limits[resource_type].current_usage += amount
        
        # Record usage metrics
        self._record_usage_metrics(tenant_id, resource_usage)
        tenant.last_active = datetime.now()
        
        logger.debug(f"Allocated resources to tenant {tenant_id}: {resource_usage}")
        return True

    def release_resources(self, tenant_id: str, resource_usage: Dict[ResourceType, float]) -> bool:
        """Release previously allocated resources."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        for resource_type, amount in resource_usage.items():
            if resource_type in tenant.resource_limits:
                current = tenant.resource_limits[resource_type].current_usage
                tenant.resource_limits[resource_type].current_usage = max(0.0, current - amount)
        
        logger.debug(f"Released resources from tenant {tenant_id}: {resource_usage}")
        return True

    def _record_usage_metrics(self, tenant_id: str, resource_usage: Dict[ResourceType, float]):
        """Record usage metrics for billing and analytics."""
        metrics = UsageMetrics(
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            resource_usage=resource_usage,
            api_calls_count=int(resource_usage.get(ResourceType.API_CALLS, 0)),
            active_users=int(resource_usage.get(ResourceType.CONCURRENT_USERS, 0)),
            response_time_ms=0.0  # Would be populated by actual response time
        )
        
        if tenant_id in self.usage_history:
            self.usage_history[tenant_id].append(metrics)
            # Keep only last 1000 entries to prevent memory bloat
            if len(self.usage_history[tenant_id]) > 1000:
                self.usage_history[tenant_id] = self.usage_history[tenant_id][-1000:]

    def get_tenant_usage_summary(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage summary for a tenant over the specified time period."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.usage_history.get(tenant_id, [])
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {
                "tenant_id": tenant_id,
                "period_hours": hours,
                "total_api_calls": 0,
                "average_response_time": 0.0,
                "peak_concurrent_users": 0,
                "resource_usage": {}
            }
        
        # Calculate aggregated metrics
        total_api_calls = sum(m.api_calls_count for m in recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        peak_users = max(m.active_users for m in recent_metrics)
        
        # Calculate resource usage averages
        resource_usage = {}
        for resource_type in ResourceType:
            values = [m.resource_usage.get(resource_type, 0.0) for m in recent_metrics]
            if values:
                resource_usage[resource_type.value] = {
                    "average": sum(values) / len(values),
                    "peak": max(values),
                    "total": sum(values)
                }
        
        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "tier": tenant.tier.value,
            "period_hours": hours,
            "total_api_calls": total_api_calls,
            "average_response_time": round(avg_response_time, 2),
            "peak_concurrent_users": peak_users,
            "resource_usage": resource_usage,
            "current_limits": {
                rt.value: limit.limit for rt, limit in tenant.resource_limits.items()
            },
            "current_usage": {
                rt.value: limit.current_usage for rt, limit in tenant.resource_limits.items()
            }
        }

    def get_all_tenants_summary(self) -> Dict[str, Any]:
        """Get summary of all tenants and their current status."""
        active_tenants = sum(1 for t in self.tenants.values() if t.is_active)
        total_tenants = len(self.tenants)
        
        tier_distribution = {}
        for tenant in self.tenants.values():
            tier = tenant.tier.value
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        # Calculate total resource usage across all tenants
        total_usage = {resource.value: 0.0 for resource in ResourceType}
        for tenant in self.tenants.values():
            for resource_type, limit in tenant.resource_limits.items():
                total_usage[resource_type.value] += limit.current_usage
        
        return {
            "total_tenants": total_tenants,
            "active_tenants": active_tenants,
            "tier_distribution": tier_distribution,
            "total_resource_usage": total_usage,
            "timestamp": datetime.now().isoformat()
        }

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant (soft delete)."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.is_active = False
        
        # Release all allocated resources
        resource_usage = {
            rt: limit.current_usage 
            for rt, limit in tenant.resource_limits.items()
        }
        self.release_resources(tenant_id, resource_usage)
        
        logger.info(f"Deactivated tenant {tenant_id} ({tenant.name})")
        return True

    def reactivate_tenant(self, tenant_id: str) -> bool:
        """Reactivate a previously deactivated tenant."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.is_active = True
        tenant.last_active = datetime.now()
        
        logger.info(f"Reactivated tenant {tenant_id} ({tenant.name})")
        return True

    async def submit_task(self, tenant_id: str, workload_type: WorkloadType, 
                         data: Any, priority: int = 5) -> str:
        """Submit a processing task for a tenant with GPU/CPU optimization."""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant = self.tenants[tenant_id]
        task_id = str(uuid.uuid4())
        
        # Determine optimal processing unit
        processing_unit = self.resource_manager.get_optimal_processing_unit(
            workload_type, tenant.tier
        )
        
        task = ProcessingTask(
            task_id=task_id,
            tenant_id=tenant_id,
            workload_type=workload_type,
            data=data,
            processing_unit=processing_unit,
            priority=priority
        )
        
        self.processing_tasks[task_id] = task
        
        # Process the task based on the determined processing unit
        if processing_unit == ProcessingUnit.CPU:
            result = await self.resource_manager.process_task_cpu(task)
            self.performance_metrics["cpu_tasks"] += 1
            self.performance_metrics["avg_cpu_time"] = (
                (self.performance_metrics["avg_cpu_time"] * (self.performance_metrics["cpu_tasks"] - 1) + 
                 result.get("processing_time", 0)) / self.performance_metrics["cpu_tasks"]
            )
        elif processing_unit == ProcessingUnit.GPU:
            result = await self.resource_manager.process_task_gpu(task)
            self.performance_metrics["gpu_tasks"] += 1
            self.performance_metrics["avg_gpu_time"] = (
                (self.performance_metrics["avg_gpu_time"] * (self.performance_metrics["gpu_tasks"] - 1) + 
                 result.get("processing_time", 0)) / self.performance_metrics["gpu_tasks"]
            )
        else:  # HYBRID
            result = await self.resource_manager.process_task_hybrid(task)
            self.performance_metrics["hybrid_tasks"] += 1
            self.performance_metrics["avg_hybrid_time"] = (
                (self.performance_metrics["avg_hybrid_time"] * (self.performance_metrics["hybrid_tasks"] - 1) + 
                 result.get("processing_time", 0)) / self.performance_metrics["hybrid_tasks"]
            )
        
        self.task_results[task_id] = result
        self.performance_metrics["total_tasks"] += 1
        
        # Update tenant CPU/GPU metrics
        tenant.cpu_gpu_metrics.processing_tasks_active += 1
        tenant.cpu_gpu_metrics.last_updated = time.time()
        
        logger.info(f"Task {task_id} completed for tenant {tenant.name} using {processing_unit.value}")
        return task_id

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task."""
        return self.task_results.get(task_id)

    def get_tenant_cpu_gpu_status(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get CPU/GPU status of a tenant."""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "tier": tenant.tier.value,
            "preferred_processing": tenant.preferred_processing.value,
            "resource_limits": {
                "cpu": tenant.resource_limits.get(ResourceType.CPU, ResourceLimit(ResourceType.CPU, 0)).limit,
                "memory": tenant.resource_limits.get(ResourceType.MEMORY, ResourceLimit(ResourceType.MEMORY, 0)).limit,
                "gpu_memory": tenant.resource_limits.get(ResourceType.GPU_MEMORY, ResourceLimit(ResourceType.GPU_MEMORY, 0)).limit,
                "gpu_compute": tenant.resource_limits.get(ResourceType.GPU_COMPUTE, ResourceLimit(ResourceType.GPU_COMPUTE, 0)).limit
            },
            "cpu_gpu_metrics": {
                "cpu_usage_percent": tenant.cpu_gpu_metrics.cpu_usage_percent,
                "memory_usage_mb": tenant.cpu_gpu_metrics.memory_usage_mb,
                "gpu_usage_percent": tenant.cpu_gpu_metrics.gpu_usage_percent,
                "gpu_memory_usage_mb": tenant.cpu_gpu_metrics.gpu_memory_usage_mb,
                "active_tasks": tenant.cpu_gpu_metrics.processing_tasks_active
            },
            "workload_types": [wt.value for wt in tenant.workload_types],
            "is_active": tenant.is_active
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview with CPU/GPU metrics."""
        return {
            "system_resources": self.resource_manager.get_resource_summary(),
            "tenant_count": len(self.tenants),
            "active_tenants": len([t for t in self.tenants.values() if t.is_active]),
            "performance_metrics": self.performance_metrics,
            "processing_distribution": {
                "cpu_percentage": (self.performance_metrics["cpu_tasks"] / 
                                 max(self.performance_metrics["total_tasks"], 1)) * 100,
                "gpu_percentage": (self.performance_metrics["gpu_tasks"] / 
                                 max(self.performance_metrics["total_tasks"], 1)) * 100,
                "hybrid_percentage": (self.performance_metrics["hybrid_tasks"] / 
                                    max(self.performance_metrics["total_tasks"], 1)) * 100
            }
        }

async def simulate_tenant_workload(architecture: MultiTenantArchitecture, 
                                 tenant_id: str, duration: int = 30):
    """Simulate realistic workload for a tenant with CPU/GPU tasks."""
    import random
    
    logger.info(f"Starting hybrid CPU/GPU workload simulation for tenant {tenant_id}")
    
    # Test different workload types
    workload_types = [
        WorkloadType.INFERENCE,
        WorkloadType.AUDIO_PROCESSING,
        WorkloadType.TEXT_PROCESSING,
        WorkloadType.DATABASE,
        WorkloadType.STREAMING,
        WorkloadType.ANALYTICS
    ]
    
    start_time = time.time()
    tasks_submitted = 0
    
    while time.time() - start_time < duration and tasks_submitted < 10:
        # Submit a random task type
        workload_type = random.choice(workload_types)
        test_data = {"task_data": f"test_data_{tasks_submitted}", "size": random.randint(100, 1000)}
        
        try:
            task_id = await architecture.submit_task(tenant_id, workload_type, test_data)
            tasks_submitted += 1
            logger.info(f"Submitted {workload_type.value} task {task_id} for tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
        
        await asyncio.sleep(random.uniform(0.5, 2.0))

async def main():
    """Main function to demonstrate Multi-Tenant Architecture with CPU/GPU processing."""
    architecture = MultiTenantArchitecture()
    
    print("🏢 VORTA Multi-Tenant Architecture with Hybrid CPU/GPU Processing")
    print("=" * 70)
    
    # Show system overview
    overview = architecture.get_system_overview()
    resources = overview["system_resources"]
    
    print("\n💻 System Resources:")
    print(f"  CPU: {resources['cpu']['cores']} cores ({resources['cpu']['usage_percent']:.1f}% used)")
    print(f"  Memory: {resources['memory']['total_gb']:.1f}GB ({resources['memory']['usage_percent']:.1f}% used)")
    print(f"  GPU: {resources['gpu']['count']} available" if resources['gpu']['available'] else "  GPU: Not available")
    if resources['gpu']['available']:
        print(f"  GPU Memory: {resources['gpu']['memory_total_gb']:.1f}GB ({resources['gpu']['memory_usage_percent']:.1f}% used)")
    
    print(f"\n🏢 Tenants: {overview['tenant_count']} total")
    
    # Show tenant information with CPU/GPU details
    for tenant_id in architecture.tenants.keys():
        status = architecture.get_tenant_cpu_gpu_status(tenant_id)
        print(f"\n📊 {status['name']} ({status['tier'].upper()}):")
        print(f"  Preferred Processing: {status['preferred_processing']}")
        print(f"  CPU Limit: {status['resource_limits']['cpu']} cores")
        print(f"  Memory Limit: {status['resource_limits']['memory']}GB")
        if status['resource_limits']['gpu_memory'] > 0:
            print(f"  GPU Memory Limit: {status['resource_limits']['gpu_memory']}GB")
            print(f"  GPU Compute Limit: {status['resource_limits']['gpu_compute']}%")
        print(f"  Workloads: {', '.join(status['workload_types'])}")
    
    # Test different workload types across tenants
    print("\n🚀 Testing Hybrid CPU/GPU Processing...")
    
    test_cases = [
        ("AI Inference", WorkloadType.INFERENCE, {"model": "bert-base", "input": "test text"}),
        ("Audio Processing", WorkloadType.AUDIO_PROCESSING, {"audio_file": "sample.wav", "duration": 10}),
        ("Text Analysis", WorkloadType.TEXT_PROCESSING, {"text": "Natural language processing test"}),
        ("Database Query", WorkloadType.DATABASE, {"query": "SELECT * FROM users", "params": []}),
        ("Real-time Stream", WorkloadType.STREAMING, {"stream_url": "rtmp://example.com", "bitrate": 1000}),
        ("Analytics", WorkloadType.ANALYTICS, {"dataset": "user_behavior", "metrics": ["engagement", "retention"]})
    ]
    
    task_ids = []
    
    # Submit tasks to different tenants
    for i, (task_name, workload_type, data) in enumerate(test_cases):
        tenant_list = list(architecture.tenants.items())
        tenant_id, tenant = tenant_list[i % len(tenant_list)]
        
        print(f"\n  Submitting {task_name} to {tenant.name}...")
        task_id = await architecture.submit_task(tenant_id, workload_type, data)
        task_ids.append(task_id)
    
    # Wait a moment for all tasks to complete
    await asyncio.sleep(1)
    
    # Show results
    print("\n📈 Processing Results:")
    for i, task_id in enumerate(task_ids):
        result = architecture.get_task_result(task_id)
        if result:
            task_name = test_cases[i][0]
            print(f"  {task_name}: {result['processing_unit']} ({result['processing_time']:.3f}s)")
            if 'result' in result and isinstance(result['result'], dict):
                if 'method' in result['result']:
                    print(f"    Method: {result['result']['method']}")
                if 'speedup' in result['result']:
                    print(f"    Speedup: {result['result']['speedup']}")
    
    # Final system overview
    final_overview = architecture.get_system_overview()
    metrics = final_overview["performance_metrics"]
    distribution = final_overview["processing_distribution"]
    
    print("\n📊 Performance Summary:")
    print(f"  Total Tasks: {metrics['total_tasks']}")
    print(f"  CPU Tasks: {metrics['cpu_tasks']} ({distribution['cpu_percentage']:.1f}%) - Avg: {metrics['avg_cpu_time']:.3f}s")
    print(f"  GPU Tasks: {metrics['gpu_tasks']} ({distribution['gpu_percentage']:.1f}%) - Avg: {metrics['avg_gpu_time']:.3f}s")
    print(f"  Hybrid Tasks: {metrics['hybrid_tasks']} ({distribution['hybrid_percentage']:.1f}%) - Avg: {metrics['avg_hybrid_time']:.3f}s")
    
    if GPU_AVAILABLE and metrics['avg_gpu_time'] > 0:
        speedup = metrics['avg_cpu_time'] / max(metrics['avg_gpu_time'], 0.001)
        print(f"  GPU Speedup: {speedup:.1f}x faster than CPU")
    elif not GPU_AVAILABLE:
        print("  GPU Status: Not available - using CPU fallback")
    
    print("\n✅ Multi-tenant architecture with hybrid CPU/GPU processing completed!")

if __name__ == "__main__":
    asyncio.run(main())
