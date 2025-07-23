# services/global_deployment/multi_tenant_architecture.py
"""
VORTA AGI: Multi-Tenant Architecture System

Enterprise customer isolation and scaling for global deployment
- Tenant isolation and resource management
- Per-tenant configuration and customization
- Scalable tenant provisioning
- Resource quotas and billing integration
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TenantTier(Enum):
    """Different tenant service tiers."""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class TenantStatus(Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PROVISIONING = "provisioning"
    DEPROVISIONING = "deprovisioning"
    TRIAL = "trial"

@dataclass
class ResourceQuotas:
    """Resource quotas for a tenant."""
    max_concurrent_users: int = 100
    max_api_calls_per_hour: int = 10000
    max_storage_gb: int = 50
    max_audio_minutes_per_month: int = 1000
    max_voice_models: int = 5
    max_custom_integrations: int = 3
    priority_support: bool = False
    advanced_analytics: bool = False
    white_label_branding: bool = False

@dataclass
class TenantConfiguration:
    """Per-tenant configuration settings."""
    tenant_id: str
    name: str
    tier: TenantTier
    status: TenantStatus
    created_at: datetime
    quotas: ResourceQuotas
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    api_keys: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    contact_info: Dict[str, str] = field(default_factory=dict)
    billing_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TenantUsageStats:
    """Current usage statistics for a tenant."""
    tenant_id: str
    current_concurrent_users: int = 0
    api_calls_this_hour: int = 0
    storage_used_gb: float = 0.0
    audio_minutes_this_month: int = 0
    voice_models_created: int = 0
    custom_integrations: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    monthly_stats: Dict[str, Any] = field(default_factory=dict)

class MultiTenantArchitecture:
    """
    Multi-tenant architecture system for enterprise customer isolation.
    
    Features:
    - Tenant provisioning and deprovisioning
    - Resource isolation and quotas
    - Per-tenant configuration management
    - Usage monitoring and billing support
    - Scalable tenant infrastructure
    """
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfiguration] = {}
        self.usage_stats: Dict[str, TenantUsageStats] = {}
        self.active_sessions: Dict[str, Set[str]] = {}  # tenant_id -> session_ids
        self.tier_quotas = self._initialize_tier_quotas()
        
    def _initialize_tier_quotas(self) -> Dict[TenantTier, ResourceQuotas]:
        """Initialize default resource quotas for each tier."""
        return {
            TenantTier.STARTER: ResourceQuotas(
                max_concurrent_users=50,
                max_api_calls_per_hour=5000,
                max_storage_gb=10,
                max_audio_minutes_per_month=500,
                max_voice_models=2,
                max_custom_integrations=1,
                priority_support=False,
                advanced_analytics=False,
                white_label_branding=False
            ),
            TenantTier.PROFESSIONAL: ResourceQuotas(
                max_concurrent_users=200,
                max_api_calls_per_hour=25000,
                max_storage_gb=100,
                max_audio_minutes_per_month=5000,
                max_voice_models=10,
                max_custom_integrations=5,
                priority_support=True,
                advanced_analytics=True,
                white_label_branding=False
            ),
            TenantTier.ENTERPRISE: ResourceQuotas(
                max_concurrent_users=1000,
                max_api_calls_per_hour=100000,
                max_storage_gb=500,
                max_audio_minutes_per_month=25000,
                max_voice_models=50,
                max_custom_integrations=20,
                priority_support=True,
                advanced_analytics=True,
                white_label_branding=True
            ),
            TenantTier.ENTERPRISE_PLUS: ResourceQuotas(
                max_concurrent_users=5000,
                max_api_calls_per_hour=500000,
                max_storage_gb=2000,
                max_audio_minutes_per_month=100000,
                max_voice_models=200,
                max_custom_integrations=100,
                priority_support=True,
                advanced_analytics=True,
                white_label_branding=True
            )
        }

    async def provision_tenant(self, 
                             name: str, 
                             tier: TenantTier,
                             contact_info: Dict[str, str],
                             custom_quotas: Optional[ResourceQuotas] = None) -> TenantConfiguration:
        """
        Provision a new tenant with isolated resources.
        
        Args:
            name: Tenant organization name
            tier: Service tier
            contact_info: Primary contact information
            custom_quotas: Optional custom resource quotas
            
        Returns:
            TenantConfiguration with provisioned tenant details
        """
        try:
            tenant_id = str(uuid.uuid4())
            
            # Get quotas for tier or use custom
            quotas = custom_quotas or self.tier_quotas[tier]
            
            # Create tenant configuration
            tenant_config = TenantConfiguration(
                tenant_id=tenant_id,
                name=name,
                tier=tier,
                status=TenantStatus.PROVISIONING,
                created_at=datetime.now(),
                quotas=quotas,
                contact_info=contact_info,
                api_keys=[self._generate_api_key()],
                domains=[],
                custom_settings={},
                billing_info={}
            )
            
            # Initialize usage stats
            usage_stats = TenantUsageStats(tenant_id=tenant_id)
            
            # Provision tenant infrastructure
            await self._provision_tenant_infrastructure(tenant_config)
            
            # Store tenant data
            self.tenants[tenant_id] = tenant_config
            self.usage_stats[tenant_id] = usage_stats
            self.active_sessions[tenant_id] = set()
            
            # Update status to active
            tenant_config.status = TenantStatus.ACTIVE
            
            logger.info(f"Provisioned tenant '{name}' ({tenant_id}) with tier {tier.value}")
            return tenant_config
            
        except Exception as e:
            logger.error(f"Failed to provision tenant '{name}': {e}")
            raise TenantProvisioningError(f"Tenant provisioning failed: {e}")

    async def _provision_tenant_infrastructure(self, tenant_config: TenantConfiguration):
        """Provision isolated infrastructure for the tenant."""
        try:
            # Simulate infrastructure provisioning
            await asyncio.sleep(0.1)  # Simulate deployment time
            
            # In production, this would:
            # - Create isolated database schemas
            # - Set up tenant-specific storage buckets
            # - Configure network isolation
            # - Deploy tenant-specific service instances
            # - Set up monitoring and logging
            
            infrastructure_config = {
                "database_schema": f"tenant_{tenant_config.tenant_id.replace('-', '_')}",
                "storage_bucket": f"vorta-tenant-{tenant_config.tenant_id}",
                "api_endpoint": f"https://{tenant_config.tenant_id}.api.vorta.ai",
                "monitoring_namespace": f"tenant-{tenant_config.tenant_id}",
                "isolation_level": "full"
            }
            
            tenant_config.custom_settings["infrastructure"] = infrastructure_config
            
            logger.info(f"Provisioned infrastructure for tenant {tenant_config.tenant_id}")
            
        except Exception as e:
            raise TenantProvisioningError(f"Infrastructure provisioning failed: {e}")

    def _generate_api_key(self) -> str:
        """Generate a secure API key for tenant."""
        return f"vk_{uuid.uuid4().hex[:16]}"

    async def check_resource_limits(self, tenant_id: str, resource_type: str, 
                                  requested_amount: int = 1) -> Dict[str, Any]:
        """
        Check if tenant can use requested resources without exceeding quotas.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource (users, api_calls, storage, etc.)
            requested_amount: Amount of resource requested
            
        Returns:
            Dict with check result and quota information
        """
        try:
            if tenant_id not in self.tenants:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")
            
            tenant = self.tenants[tenant_id]
            usage = self.usage_stats[tenant_id]
            quotas = tenant.quotas
            
            check_result = {
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "requested_amount": requested_amount,
                "allowed": False,
                "current_usage": 0,
                "quota_limit": 0,
                "remaining": 0,
                "reason": ""
            }
            
            # Check different resource types
            if resource_type == "concurrent_users":
                current = usage.current_concurrent_users
                limit = quotas.max_concurrent_users
                check_result["current_usage"] = current
                check_result["quota_limit"] = limit
                check_result["remaining"] = max(0, limit - current)
                
                if current + requested_amount <= limit:
                    check_result["allowed"] = True
                else:
                    check_result["reason"] = f"Would exceed concurrent user limit ({limit})"
                    
            elif resource_type == "api_calls":
                current = usage.api_calls_this_hour
                limit = quotas.max_api_calls_per_hour
                check_result["current_usage"] = current
                check_result["quota_limit"] = limit
                check_result["remaining"] = max(0, limit - current)
                
                if current + requested_amount <= limit:
                    check_result["allowed"] = True
                else:
                    check_result["reason"] = f"Would exceed API call limit ({limit}/hour)"
                    
            elif resource_type == "storage":
                current = usage.storage_used_gb
                limit = quotas.max_storage_gb
                check_result["current_usage"] = current
                check_result["quota_limit"] = limit
                check_result["remaining"] = max(0, limit - current)
                
                if current + requested_amount <= limit:
                    check_result["allowed"] = True
                else:
                    check_result["reason"] = f"Would exceed storage limit ({limit} GB)"
                    
            elif resource_type == "audio_minutes":
                current = usage.audio_minutes_this_month
                limit = quotas.max_audio_minutes_per_month
                check_result["current_usage"] = current
                check_result["quota_limit"] = limit
                check_result["remaining"] = max(0, limit - current)
                
                if current + requested_amount <= limit:
                    check_result["allowed"] = True
                else:
                    check_result["reason"] = f"Would exceed audio processing limit ({limit} min/month)"
                    
            elif resource_type == "voice_models":
                current = usage.voice_models_created
                limit = quotas.max_voice_models
                check_result["current_usage"] = current
                check_result["quota_limit"] = limit
                check_result["remaining"] = max(0, limit - current)
                
                if current + requested_amount <= limit:
                    check_result["allowed"] = True
                else:
                    check_result["reason"] = f"Would exceed voice model limit ({limit})"
                    
            else:
                check_result["reason"] = f"Unknown resource type: {resource_type}"
            
            return check_result
            
        except Exception as e:
            logger.error(f"Resource limit check failed for {tenant_id}: {e}")
            return {
                "allowed": False,
                "reason": f"Error checking limits: {e}",
                "error": True
            }

    async def update_tenant_usage(self, tenant_id: str, resource_type: str, 
                                amount: int, operation: str = "increment") -> bool:
        """
        Update tenant resource usage statistics.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource being updated
            amount: Amount to add/subtract
            operation: 'increment' or 'decrement' or 'set'
            
        Returns:
            True if update successful
        """
        try:
            if tenant_id not in self.usage_stats:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")
                
            usage = self.usage_stats[tenant_id]
            
            # Update based on resource type and operation
            if resource_type == "concurrent_users":
                if operation == "increment":
                    usage.current_concurrent_users += amount
                elif operation == "decrement":
                    usage.current_concurrent_users = max(0, usage.current_concurrent_users - amount)
                elif operation == "set":
                    usage.current_concurrent_users = amount
                    
            elif resource_type == "api_calls":
                if operation == "increment":
                    usage.api_calls_this_hour += amount
                elif operation == "set":
                    usage.api_calls_this_hour = amount
                    
            elif resource_type == "storage":
                if operation == "increment":
                    usage.storage_used_gb += amount
                elif operation == "decrement":
                    usage.storage_used_gb = max(0, usage.storage_used_gb - amount)
                elif operation == "set":
                    usage.storage_used_gb = amount
                    
            elif resource_type == "audio_minutes":
                if operation == "increment":
                    usage.audio_minutes_this_month += amount
                elif operation == "set":
                    usage.audio_minutes_this_month = amount
                    
            elif resource_type == "voice_models":
                if operation == "increment":
                    usage.voice_models_created += amount
                elif operation == "decrement":
                    usage.voice_models_created = max(0, usage.voice_models_created - amount)
                elif operation == "set":
                    usage.voice_models_created = amount
            
            # Update last activity
            usage.last_activity = datetime.now()
            
            logger.debug(f"Updated {resource_type} for {tenant_id}: {operation} {amount}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update usage for {tenant_id}: {e}")
            return False

    async def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a tenant."""
        try:
            if tenant_id not in self.tenants:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")
                
            tenant = self.tenants[tenant_id]
            usage = self.usage_stats[tenant_id]
            
            # Calculate utilization percentages
            utilization = {}
            quotas_dict = {
                "concurrent_users": (usage.current_concurrent_users, tenant.quotas.max_concurrent_users),
                "api_calls": (usage.api_calls_this_hour, tenant.quotas.max_api_calls_per_hour),
                "storage": (usage.storage_used_gb, tenant.quotas.max_storage_gb),
                "audio_minutes": (usage.audio_minutes_this_month, tenant.quotas.max_audio_minutes_per_month),
                "voice_models": (usage.voice_models_created, tenant.quotas.max_voice_models)
            }
            
            for resource, (current, limit) in quotas_dict.items():
                utilization[resource] = {
                    "current": current,
                    "limit": limit,
                    "percentage": (current / limit * 100) if limit > 0 else 0,
                    "remaining": max(0, limit - current)
                }
            
            dashboard = {
                "tenant_info": {
                    "tenant_id": tenant_id,
                    "name": tenant.name,
                    "tier": tenant.tier.value,
                    "status": tenant.status.value,
                    "created_at": tenant.created_at.isoformat(),
                    "api_keys_count": len(tenant.api_keys),
                    "domains_count": len(tenant.domains)
                },
                "resource_utilization": utilization,
                "current_activity": {
                    "active_sessions": len(self.active_sessions.get(tenant_id, set())),
                    "last_activity": usage.last_activity.isoformat(),
                    "monthly_stats": usage.monthly_stats
                },
                "features": {
                    "priority_support": tenant.quotas.priority_support,
                    "advanced_analytics": tenant.quotas.advanced_analytics,
                    "white_label_branding": tenant.quotas.white_label_branding
                },
                "health_status": await self._assess_tenant_health(tenant_id)
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard for {tenant_id}: {e}")
            return {"error": str(e)}

    async def _assess_tenant_health(self, tenant_id: str) -> Dict[str, Any]:
        """Assess tenant health based on usage patterns and limits."""
        try:
            tenant = self.tenants[tenant_id]
            usage = self.usage_stats[tenant_id]
            
            health_score = 100
            issues = []
            warnings = []
            
            # Check resource utilization
            checks = [
                ("concurrent_users", usage.current_concurrent_users, tenant.quotas.max_concurrent_users),
                ("api_calls", usage.api_calls_this_hour, tenant.quotas.max_api_calls_per_hour),
                ("storage", usage.storage_used_gb, tenant.quotas.max_storage_gb),
                ("audio_minutes", usage.audio_minutes_this_month, tenant.quotas.max_audio_minutes_per_month)
            ]
            
            for resource, current, limit in checks:
                if limit > 0:
                    utilization = (current / limit) * 100
                    
                    if utilization >= 95:
                        issues.append(f"{resource} at {utilization:.1f}% capacity")
                        health_score -= 20
                    elif utilization >= 80:
                        warnings.append(f"{resource} at {utilization:.1f}% capacity")
                        health_score -= 5
            
            # Check last activity
            hours_since_activity = (datetime.now() - usage.last_activity).total_seconds() / 3600
            if hours_since_activity > 24:
                warnings.append(f"No activity for {hours_since_activity:.1f} hours")
                health_score -= 5
            
            # Determine overall health
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 60:
                health_status = "warning"
            else:
                health_status = "critical"
            
            return {
                "health_score": max(0, health_score),
                "status": health_status,
                "issues": issues,
                "warnings": warnings,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health assessment failed for {tenant_id}: {e}")
            return {
                "health_score": 0,
                "status": "error",
                "issues": [f"Health check failed: {e}"]
            }

    async def get_all_tenants_summary(self) -> Dict[str, Any]:
        """Get summary of all tenants for admin dashboard."""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_tenants": len(self.tenants),
                "tenants_by_tier": {},
                "tenants_by_status": {},
                "resource_usage": {
                    "total_concurrent_users": 0,
                    "total_api_calls_hour": 0,
                    "total_storage_gb": 0,
                    "total_audio_minutes_month": 0
                },
                "tenant_list": []
            }
            
            # Initialize counters
            for tier in TenantTier:
                summary["tenants_by_tier"][tier.value] = 0
                
            for status in TenantStatus:
                summary["tenants_by_status"][status.value] = 0
            
            # Process each tenant
            for tenant_id, tenant in self.tenants.items():
                usage = self.usage_stats.get(tenant_id, TenantUsageStats(tenant_id))
                
                # Count by tier and status
                summary["tenants_by_tier"][tenant.tier.value] += 1
                summary["tenants_by_status"][tenant.status.value] += 1
                
                # Aggregate resource usage
                summary["resource_usage"]["total_concurrent_users"] += usage.current_concurrent_users
                summary["resource_usage"]["total_api_calls_hour"] += usage.api_calls_this_hour
                summary["resource_usage"]["total_storage_gb"] += usage.storage_used_gb
                summary["resource_usage"]["total_audio_minutes_month"] += usage.audio_minutes_this_month
                
                # Add tenant summary
                tenant_summary = {
                    "tenant_id": tenant_id,
                    "name": tenant.name,
                    "tier": tenant.tier.value,
                    "status": tenant.status.value,
                    "created_at": tenant.created_at.isoformat(),
                    "current_users": usage.current_concurrent_users,
                    "last_activity": usage.last_activity.isoformat()
                }
                summary["tenant_list"].append(tenant_summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate tenants summary: {e}")
            return {"error": str(e)}

# Custom exceptions
class TenantError(Exception):
    """Base exception for tenant-related errors."""
    pass

class TenantNotFoundError(TenantError):
    """Raised when tenant is not found."""
    pass

class TenantProvisioningError(TenantError):
    """Raised when tenant provisioning fails."""
    pass

class ResourceQuotaExceededError(TenantError):
    """Raised when resource quota is exceeded."""
    pass

# Usage example and testing
async def test_multi_tenant_architecture():
    """Test the Multi-Tenant Architecture functionality."""
    mta = MultiTenantArchitecture()
    
    print("🏢 Testing Multi-Tenant Architecture...")
    
    # Test tenant provisioning
    print("\n📋 Provisioning test tenants...")
    
    tenants_to_create = [
        ("Acme Corp", TenantTier.ENTERPRISE, {"email": "admin@acme.com", "phone": "+1-555-0101"}),
        ("StartupAI Inc", TenantTier.PROFESSIONAL, {"email": "cto@startup.ai", "phone": "+1-555-0102"}),
        ("SmallBiz LLC", TenantTier.STARTER, {"email": "owner@smallbiz.com", "phone": "+1-555-0103"})
    ]
    
    created_tenants = []
    for name, tier, contact in tenants_to_create:
        tenant = await mta.provision_tenant(name, tier, contact)
        created_tenants.append(tenant)
        print(f"  ✅ Created: {name} ({tier.value}) - ID: {tenant.tenant_id[:8]}...")
    
    # Test resource limit checking
    print("\n🔍 Testing resource limit checks...")
    for tenant in created_tenants[:2]:  # Test first two tenants
        check = await mta.check_resource_limits(tenant.tenant_id, "concurrent_users", 25)
        print(f"  {tenant.name}: {check['resource_type']} - {'✅ Allowed' if check['allowed'] else '❌ Denied'}")
        print(f"    Usage: {check['current_usage']}/{check['quota_limit']} (remaining: {check['remaining']})")
    
    # Test usage updates
    print("\n📊 Testing usage updates...")
    test_tenant = created_tenants[0]
    await mta.update_tenant_usage(test_tenant.tenant_id, "concurrent_users", 50, "increment")
    await mta.update_tenant_usage(test_tenant.tenant_id, "api_calls", 5000, "increment")
    await mta.update_tenant_usage(test_tenant.tenant_id, "storage", 25.5, "set")
    print(f"  Updated usage for {test_tenant.name}")
    
    # Test tenant dashboard
    print("\n📈 Testing tenant dashboard...")
    dashboard = await mta.get_tenant_dashboard(test_tenant.tenant_id)
    print(f"  Dashboard for {dashboard['tenant_info']['name']}:")
    print(f"    Status: {dashboard['tenant_info']['status']}")
    print(f"    Health: {dashboard['health_status']['status']} ({dashboard['health_status']['health_score']})")
    
    for resource, data in dashboard['resource_utilization'].items():
        print(f"    {resource}: {data['current']}/{data['limit']} ({data['percentage']:.1f}%)")
    
    # Test admin summary
    print("\n👥 Testing admin summary...")
    summary = await mta.get_all_tenants_summary()
    print(f"  Total tenants: {summary['total_tenants']}")
    print("  By tier:")
    for tier, count in summary['tenants_by_tier'].items():
        if count > 0:
            print(f"    {tier}: {count}")
    print(f"  Total concurrent users: {summary['resource_usage']['total_concurrent_users']}")

if __name__ == "__main__":
    asyncio.run(test_multi_tenant_architecture())
