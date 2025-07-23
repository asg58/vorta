# services/global_deployment/api_marketplace.py
"""
VORTA AGI: API Marketplace & Third-Party Integration Ecosystem

Enterprise API marketplace for third-party integrations and custom extensions
- API marketplace with plugin discovery
- Third-party developer portal
- Integration management and deployment
- Revenue sharing and billing
- API versioning and compatibility
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

class IntegrationStatus(Enum):
    """Status of integrations in the marketplace."""
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"

class IntegrationCategory(Enum):
    """Categories for marketplace integrations."""
    VOICE_PROCESSING = "voice_processing"
    AI_MODELS = "ai_models"
    ANALYTICS = "analytics"
    CRM_SYSTEMS = "crm_systems"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    SECURITY = "security"
    CUSTOM_WORKFLOWS = "custom_workflows"

class RevenueModel(Enum):
    """Revenue sharing models."""
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    REVENUE_SHARE = "revenue_share"

@dataclass
class Developer:
    """Third-party developer information."""
    developer_id: str
    name: str
    email: str
    company: Optional[str] = None
    verified: bool = False
    api_key: str = field(default_factory=lambda: f"dev_{uuid.uuid4().hex[:16]}")
    created_at: datetime = field(default_factory=datetime.now)
    total_integrations: int = 0
    total_installs: int = 0

@dataclass
class Integration:
    """Marketplace integration/plugin."""
    integration_id: str
    name: str
    description: str
    developer_id: str
    category: IntegrationCategory
    version: str
    status: IntegrationStatus
    pricing: Dict[str, Any]
    requirements: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    documentation_url: str = ""
    source_code_url: str = ""
    install_count: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Installation:
    """Integration installation record."""
    installation_id: str
    tenant_id: str
    integration_id: str
    installed_at: datetime
    configuration: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    usage_stats: Dict[str, int] = field(default_factory=dict)

class APIMarketplace:
    """
    API Marketplace & Third-Party Integration Ecosystem.
    
    Features:
    - Developer portal and registration
    - Integration publishing and discovery
    - Installation and configuration management
    - Usage tracking and billing
    - Review and rating system
    """
    
    def __init__(self):
        self.developers: Dict[str, Developer] = {}
        self.integrations: Dict[str, Integration] = {}
        self.installations: Dict[str, Installation] = {}
        self.featured_integrations: List[str] = []
        self.integration_analytics: Dict[str, Dict[str, Any]] = {}
        
    async def register_developer(self, 
                               name: str, 
                               email: str, 
                               company: Optional[str] = None) -> Developer:
        """
        Register a new third-party developer.
        
        Args:
            name: Developer name
            email: Contact email
            company: Optional company name
            
        Returns:
            Developer instance with API credentials
        """
        try:
            developer_id = str(uuid.uuid4())
            
            developer = Developer(
                developer_id=developer_id,
                name=name,
                email=email,
                company=company
            )
            
            self.developers[developer_id] = developer
            
            logger.info(f"Registered developer: {name} ({developer_id})")
            return developer
            
        except Exception as e:
            logger.error(f"Failed to register developer {name}: {e}")
            raise

    async def submit_integration(self, 
                               developer_id: str,
                               integration_data: Dict[str, Any]) -> Integration:
        """
        Submit a new integration to the marketplace.
        
        Args:
            developer_id: Developer's unique ID
            integration_data: Integration details and metadata
            
        Returns:
            Integration instance
        """
        try:
            if developer_id not in self.developers:
                raise ValueError(f"Developer {developer_id} not found")
            
            integration_id = str(uuid.uuid4())
            
            integration = Integration(
                integration_id=integration_id,
                name=integration_data["name"],
                description=integration_data["description"],
                developer_id=developer_id,
                category=IntegrationCategory(integration_data["category"]),
                version=integration_data.get("version", "1.0.0"),
                status=IntegrationStatus.PENDING_REVIEW,
                pricing=integration_data.get("pricing", {"model": "free"}),
                requirements=integration_data.get("requirements", {}),
                api_endpoints=integration_data.get("api_endpoints", []),
                documentation_url=integration_data.get("documentation_url", ""),
                source_code_url=integration_data.get("source_code_url", "")
            )
            
            self.integrations[integration_id] = integration
            self.developers[developer_id].total_integrations += 1
            
            # Initialize analytics
            self.integration_analytics[integration_id] = {
                "views": 0,
                "installs": 0,
                "api_calls": 0,
                "revenue": 0.0
            }
            
            logger.info(f"Submitted integration: {integration.name} by {developer_id}")
            return integration
            
        except Exception as e:
            logger.error(f"Failed to submit integration: {e}")
            raise

    async def review_integration(self, 
                               integration_id: str, 
                               approved: bool,
                               review_notes: str = "") -> bool:
        """
        Review and approve/reject an integration.
        
        Args:
            integration_id: Integration to review
            approved: Whether to approve the integration
            review_notes: Optional review feedback
            
        Returns:
            True if review was successful
        """
        try:
            if integration_id not in self.integrations:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = self.integrations[integration_id]
            
            if approved:
                integration.status = IntegrationStatus.APPROVED
                logger.info(f"Approved integration: {integration.name}")
            else:
                integration.status = IntegrationStatus.SUSPENDED
                logger.info(f"Rejected integration: {integration.name} - {review_notes}")
            
            integration.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Failed to review integration {integration_id}: {e}")
            return False

    async def publish_integration(self, integration_id: str) -> bool:
        """
        Publish an approved integration to the marketplace.
        
        Args:
            integration_id: Integration to publish
            
        Returns:
            True if published successfully
        """
        try:
            if integration_id not in self.integrations:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = self.integrations[integration_id]
            
            if integration.status != IntegrationStatus.APPROVED:
                raise ValueError(f"Integration {integration_id} is not approved for publishing")
            
            integration.status = IntegrationStatus.PUBLISHED
            integration.updated_at = datetime.now()
            
            logger.info(f"Published integration: {integration.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish integration {integration_id}: {e}")
            return False

    async def discover_integrations(self, 
                                  category: Optional[IntegrationCategory] = None,
                                  search_query: Optional[str] = None,
                                  limit: int = 20) -> List[Integration]:
        """
        Discover integrations in the marketplace.
        
        Args:
            category: Optional category filter
            search_query: Optional search query
            limit: Maximum number of results
            
        Returns:
            List of matching integrations
        """
        try:
            # Get published integrations
            published_integrations = [
                integration for integration in self.integrations.values()
                if integration.status == IntegrationStatus.PUBLISHED
            ]
            
            # Apply category filter
            if category:
                published_integrations = [
                    integration for integration in published_integrations
                    if integration.category == category
                ]
            
            # Apply search filter
            if search_query:
                search_lower = search_query.lower()
                published_integrations = [
                    integration for integration in published_integrations
                    if (search_lower in integration.name.lower() or
                        search_lower in integration.description.lower())
                ]
            
            # Sort by popularity (install count + rating)
            published_integrations.sort(
                key=lambda x: x.install_count * x.rating, 
                reverse=True
            )
            
            # Track views for analytics
            for integration in published_integrations[:limit]:
                self.integration_analytics[integration.integration_id]["views"] += 1
            
            return published_integrations[:limit]
            
        except Exception as e:
            logger.error(f"Integration discovery failed: {e}")
            return []

    async def install_integration(self, 
                                tenant_id: str, 
                                integration_id: str,
                                configuration: Optional[Dict[str, Any]] = None) -> Installation:
        """
        Install an integration for a tenant.
        
        Args:
            tenant_id: Tenant installing the integration
            integration_id: Integration to install
            configuration: Optional configuration parameters
            
        Returns:
            Installation record
        """
        try:
            if integration_id not in self.integrations:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = self.integrations[integration_id]
            
            if integration.status != IntegrationStatus.PUBLISHED:
                raise ValueError(f"Integration {integration_id} is not available for installation")
            
            installation_id = str(uuid.uuid4())
            
            installation = Installation(
                installation_id=installation_id,
                tenant_id=tenant_id,
                integration_id=integration_id,
                installed_at=datetime.now(),
                configuration=configuration or {}
            )
            
            self.installations[installation_id] = installation
            
            # Update integration statistics
            integration.install_count += 1
            self.integration_analytics[integration_id]["installs"] += 1
            
            # Update developer statistics
            developer = self.developers[integration.developer_id]
            developer.total_installs += 1
            
            logger.info(f"Installed integration {integration.name} for tenant {tenant_id}")
            return installation
            
        except Exception as e:
            logger.error(f"Failed to install integration {integration_id}: {e}")
            raise

    async def get_tenant_integrations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get all integrations installed by a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of installed integrations with details
        """
        try:
            tenant_installations = [
                installation for installation in self.installations.values()
                if installation.tenant_id == tenant_id and installation.status == "active"
            ]
            
            integration_details = []
            for installation in tenant_installations:
                integration = self.integrations.get(installation.integration_id)
                if integration:
                    details = {
                        "installation_id": installation.installation_id,
                        "integration": {
                            "id": integration.integration_id,
                            "name": integration.name,
                            "description": integration.description,
                            "version": integration.version,
                            "category": integration.category.value
                        },
                        "installed_at": installation.installed_at.isoformat(),
                        "configuration": installation.configuration,
                        "usage_stats": installation.usage_stats
                    }
                    integration_details.append(details)
            
            return integration_details
            
        except Exception as e:
            logger.error(f"Failed to get tenant integrations for {tenant_id}: {e}")
            return []

    async def track_integration_usage(self, 
                                    installation_id: str, 
                                    api_calls: int = 1,
                                    data_processed: float = 0.0) -> bool:
        """
        Track usage statistics for an integration installation.
        
        Args:
            installation_id: Installation ID
            api_calls: Number of API calls
            data_processed: Amount of data processed
            
        Returns:
            True if tracking successful
        """
        try:
            if installation_id not in self.installations:
                logger.warning(f"Installation {installation_id} not found")
                return False
            
            installation = self.installations[installation_id]
            integration_id = installation.integration_id
            
            # Update installation usage stats
            installation.usage_stats["total_api_calls"] = (
                installation.usage_stats.get("total_api_calls", 0) + api_calls
            )
            installation.usage_stats["total_data_processed"] = (
                installation.usage_stats.get("total_data_processed", 0.0) + data_processed
            )
            installation.usage_stats["last_used"] = datetime.now().isoformat()
            
            # Update integration analytics
            if integration_id in self.integration_analytics:
                self.integration_analytics[integration_id]["api_calls"] += api_calls
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track usage for installation {installation_id}: {e}")
            return False

    async def get_marketplace_analytics(self) -> Dict[str, Any]:
        """Get comprehensive marketplace analytics."""
        try:
            analytics = {
                "timestamp": datetime.now().isoformat(),
                "total_developers": len(self.developers),
                "total_integrations": len(self.integrations),
                "total_installations": len(self.installations),
                "integrations_by_status": {},
                "integrations_by_category": {},
                "top_integrations": [],
                "developer_leaderboard": [],
                "revenue_summary": {
                    "total_revenue": 0.0,
                    "monthly_revenue": 0.0
                }
            }
            
            # Count by status
            for status in IntegrationStatus:
                count = sum(1 for i in self.integrations.values() if i.status == status)
                analytics["integrations_by_status"][status.value] = count
            
            # Count by category
            for category in IntegrationCategory:
                count = sum(1 for i in self.integrations.values() if i.category == category)
                analytics["integrations_by_category"][category.value] = count
            
            # Top integrations by install count
            top_integrations = sorted(
                [i for i in self.integrations.values() if i.status == IntegrationStatus.PUBLISHED],
                key=lambda x: x.install_count,
                reverse=True
            )[:10]
            
            analytics["top_integrations"] = [
                {
                    "name": integration.name,
                    "developer": self.developers[integration.developer_id].name,
                    "installs": integration.install_count,
                    "rating": integration.rating,
                    "category": integration.category.value
                }
                for integration in top_integrations
            ]
            
            # Developer leaderboard
            developer_leaderboard = sorted(
                self.developers.values(),
                key=lambda x: x.total_installs,
                reverse=True
            )[:10]
            
            analytics["developer_leaderboard"] = [
                {
                    "name": dev.name,
                    "company": dev.company,
                    "total_integrations": dev.total_integrations,
                    "total_installs": dev.total_installs,
                    "verified": dev.verified
                }
                for dev in developer_leaderboard
            ]
            
            # Calculate revenue (simplified)
            total_revenue = sum(
                self.integration_analytics[iid].get("revenue", 0.0)
                for iid in self.integration_analytics
            )
            analytics["revenue_summary"]["total_revenue"] = total_revenue
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate marketplace analytics: {e}")
            return {"error": str(e)}

# Usage example and testing
async def test_api_marketplace():
    """Test the API Marketplace functionality."""
    marketplace = APIMarketplace()
    
    print("🏪 Testing API Marketplace...")
    
    # Register developers
    print("\n👨‍💻 Registering developers...")
    developers = []
    developer_data = [
        ("Alice Johnson", "alice@techcorp.com", "TechCorp AI"),
        ("Bob Smith", "bob@voicetools.io", "VoiceTools Inc"),
        ("Carol Davis", "carol@freelancer.com", None)
    ]
    
    for name, email, company in developer_data:
        dev = await marketplace.register_developer(name, email, company)
        developers.append(dev)
        print(f"  ✅ Registered: {name} - API Key: {dev.api_key}")
    
    # Submit integrations
    print("\n📦 Submitting integrations...")
    integrations_data = [
        {
            "name": "Advanced Voice Analytics",
            "description": "Real-time voice sentiment and emotion analysis",
            "category": "voice_processing",
            "version": "2.1.0",
            "pricing": {"model": "subscription", "price": 99.00}
        },
        {
            "name": "CRM Connector Pro",
            "description": "Seamless integration with major CRM systems",
            "category": "crm_systems",
            "version": "1.5.2",
            "pricing": {"model": "usage_based", "price_per_call": 0.05}
        },
        {
            "name": "Custom Workflow Builder",
            "description": "Build custom AI-powered workflows",
            "category": "custom_workflows",
            "version": "1.0.0",
            "pricing": {"model": "free"}
        }
    ]
    
    submitted_integrations = []
    for i, integration_data in enumerate(integrations_data):
        integration = await marketplace.submit_integration(developers[i].developer_id, integration_data)
        submitted_integrations.append(integration)
        print(f"  ✅ Submitted: {integration.name} ({integration.status.value})")
    
    # Review and publish integrations
    print("\n✅ Reviewing integrations...")
    for integration in submitted_integrations:
        await marketplace.review_integration(integration.integration_id, True, "Approved after review")
        await marketplace.publish_integration(integration.integration_id)
        print(f"  ✅ Published: {integration.name}")
    
    # Discover integrations
    print("\n🔍 Discovering integrations...")
    all_integrations = await marketplace.discover_integrations()
    print(f"  Found {len(all_integrations)} published integrations:")
    for integration in all_integrations:
        print(f"    • {integration.name} - {integration.category.value}")
    
    # Install integrations for test tenant
    print("\n📥 Installing integrations for test tenant...")
    tenant_id = "test_tenant_001"
    
    for integration in submitted_integrations[:2]:  # Install first 2
        installation = await marketplace.install_integration(
            tenant_id, 
            integration.integration_id,
            {"api_key": "test_key", "enabled": True}
        )
        print(f"  ✅ Installed: {integration.name} - ID: {installation.installation_id[:8]}...")
    
    # Track usage
    print("\n📊 Tracking integration usage...")
    tenant_integrations = await marketplace.get_tenant_integrations(tenant_id)
    for integration_detail in tenant_integrations:
        await marketplace.track_integration_usage(
            integration_detail["installation_id"],
            api_calls=50,
            data_processed=1.5
        )
        print(f"  📈 Tracked usage for: {integration_detail['integration']['name']}")
    
    # Get marketplace analytics
    print("\n📈 Marketplace Analytics:")
    analytics = await marketplace.get_marketplace_analytics()
    print(f"  Total developers: {analytics['total_developers']}")
    print(f"  Total integrations: {analytics['total_integrations']}")
    print(f"  Total installations: {analytics['total_installations']}")
    print(f"  Published integrations: {analytics['integrations_by_status'].get('published', 0)}")
    
    if analytics['top_integrations']:
        print("  Top integrations:")
        for integration in analytics['top_integrations'][:3]:
            print(f"    • {integration['name']} - {integration['installs']} installs")

if __name__ == "__main__":
    asyncio.run(test_api_marketplace())
