import os
import re

# De volledige projectstructuur zoals gedefinieerd in VORTA_Project_Structure.md
# We gebruiken een multiline string om de structuur direct in het script te hebben.
PROJECT_STRUCTURE = """
vorta-platform/                              # Root project directory
├── 📁 .github/                              # GitHub workflows & templates
│   ├── workflows/                           # CI/CD pipelines
│   │   ├── ci.yml                          # Continuous Integration
│   │   ├── cd-staging.yml                  # Staging deployment
│   │   ├── cd-production.yml               # Production deployment
│   │   ├── security-scan.yml               # Security scanning
│   │   ├── performance-test.yml            # Performance benchmarks
│   │   └── dependency-update.yml           # Automated dependency updates
│   ├── ISSUE_TEMPLATE/                     # Issue templates
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── performance_issue.md
│   ├── PULL_REQUEST_TEMPLATE.md            # PR template
│   └── CODEOWNERS                          # Code ownership rules
├── 📁 docs/                                # Project documentation
│   ├── 📁 architecture/                    # System architecture docs
│   │   ├── system-overview.md              # High-level architecture
│   │   ├── data-flow.md                    # Data flow diagrams
│   │   ├── security-architecture.md        # Security design
│   │   ├── performance-targets.md          # Performance specifications
│   │   └── adr/                           # Architecture Decision Records
│   │       ├── 001-microservices-approach.md
│   │       ├── 002-kubernetes-orchestration.md
│   │       ├── 003-faiss-vector-search.md
│   │       └── 004-redis-caching-strategy.md
│   ├── 📁 api/                             # API documentation
│   │   ├── openapi.yaml                    # OpenAPI specification
│   │   ├── endpoints.md                    # API endpoints documentation
│   │   ├── authentication.md               # Auth documentation
│   │   └── rate-limiting.md               # Rate limiting policies
│   ├── 📁 deployment/                      # Deployment guides
│   │   ├── kubernetes-setup.md             # K8s deployment guide
│   │   ├── monitoring-setup.md             # Monitoring configuration
│   │   ├── security-hardening.md           # Security checklist
│   │   └── disaster-recovery.md            # DR procedures
│   ├── 📁 development/                     # Developer guides
│   │   ├── getting-started.md              # Quick start guide
│   │   ├── coding-standards.md             # Code style guide
│   │   ├── testing-strategy.md             # Testing approaches
│   │   └── debugging-guide.md              # Debugging procedures
│   └── 📁 user-guides/                     # End-user documentation
│       ├── quick-start.md                  # User quick start
│       ├── api-usage.md                    # API usage examples
│       ├── troubleshooting.md              # Common issues
│       └── best-practices.md               # Usage best practices
├── 📁 services/                            # Microservices (core application)
│   ├── 📁 api-gateway/                     # Main API gateway service
│   │   ├── 📁 src/                         # Source code
│   │   │   ├── 📁 main/                    # Main application code
│   │   │   │   ├── 📁 java/com/vorta/gateway/  # Java source (if using Java)
│   │   │   │   │   ├── 📁 config/          # Configuration classes
│   │   │   │   │   │   ├── SecurityConfig.java
│   │   │   │   │   │   ├── RouteConfig.java
│   │   │   │   │   │   └── CorsConfig.java
│   │   │   │   │   ├── 📁 controller/      # REST controllers
│   │   │   │   │   │   ├── AuthController.java
│   │   │   │   │   │   ├── HealthController.java
│   │   │   │   │   │   └── InferenceController.java
│   │   │   │   │   ├── 📁 filter/          # Custom filters
│   │   │   │   │   │   ├── AuthenticationFilter.java
│   │   │   │   │   │   ├── RateLimitFilter.java
│   │   │   │   │   │   └── LoggingFilter.java
│   │   │   │   │   ├── 📁 service/         # Business logic
│   │   │   │   │   │   ├── AuthService.java
│   │   │   │   │   │   ├── RouteService.java
│   │   │   │   │   │   └── MetricsService.java
│   │   │   │   │   ├── 📁 model/           # Data models
│   │   │   │   │   │   ├── dto/            # Data Transfer Objects
│   │   │   │   │   │   │   ├── InferenceRequest.java
│   │   │   │   │   │   │   ├── InferenceResponse.java
│   │   │   │   │   │   │   └── AuthRequest.java
│   │   │   │   │   │   └── entity/         # Entity models
│   │   │   │   │   │       ├── User.java
│   │   │   │   │   │       └── ApiKey.java
│   │   │   │   │   ├── 📁 repository/      # Data access layer
│   │   │   │   │   │   ├── UserRepository.java
│   │   │   │   │   │   └── ApiKeyRepository.java
│   │   │   │   │   └── GatewayApplication.java  # Main application class
│   │   │   │   └── 📁 resources/           # Resource files
│   │   │   │       ├── application.yml     # Application configuration
│   │   │   │       ├── application-dev.yml # Development config
│   │   │   │       ├── application-prod.yml # Production config
│   │   │   │       └── 📁 static/          # Static resources
│   │   │   └── 📁 test/                    # Test code
│   │   │       ├── 📁 java/com/vorta/gateway/
│   │   │       │   ├── 📁 integration/     # Integration tests
│   │   │       │   │   ├── AuthIntegrationTest.java
│   │   │       │   │   └── GatewayIntegrationTest.java
│   │   │       │   └── 📁 unit/            # Unit tests
│   │   │       │       ├── AuthServiceTest.java
│   │   │       │       └── RouteServiceTest.java
│   │   │       └── 📁 resources/           # Test resources
│   │   │           └── application-test.yml
│   │   ├── 📁 k8s/                         # Kubernetes manifests
│   │   │   ├── deployment.yaml             # Deployment specification
│   │   │   ├── service.yaml                # Service specification
│   │   │   ├── configmap.yaml              # Configuration
│   │   │   ├── secret.yaml                 # Secrets
│   │   │   └── ingress.yaml                # Ingress rules
│   │   ├── 📁 docker/                      # Docker configuration
│   │   │   ├── Dockerfile                  # Main dockerfile
│   │   │   ├── Dockerfile.dev              # Development dockerfile
│   │   │   └── .dockerignore               # Docker ignore rules
│   │   ├── 📁 scripts/                     # Build & deployment scripts
│   │   │   ├── build.sh                    # Build script
│   │   │   ├── test.sh                     # Test script
│   │   │   └── deploy.sh                   # Deployment script
│   │   ├── pom.xml                         # Maven configuration (if Java)
│   │   ├── requirements.txt                # Python dependencies (if Python)
│   │   ├── Makefile                        # Build automation
│   │   └── README.md                       # Service documentation
│   ├── 📁 inference-engine/               # Core AI inference service
│   │   ├── 📁 src/                         # Source code
│   │   │   ├── 📁 vorta/                   # Main package
│   │   │   │   ├── 📁 core/                # Core inference logic
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── inference_engine.py # Main inference class
│   │   │   │   │   ├── model_manager.py    # Model loading/management
│   │   │   │   │   ├── quantization.py     # Model quantization
│   │   │   │   │   └── memory_manager.py   # Memory management
│   │   │   │   ├── 📁 mining/              # Semantic mining module
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── faiss_miner.py      # FAISS-based mining
│   │   │   │   │   ├── cluster_manager.py  # Cluster management
│   │   │   │   │   ├── vector_store.py     # Vector storage
│   │   │   │   │   └── similarity_search.py # Similarity algorithms
│   │   │   │   ├── 📁 cache/               # Caching layer
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── redis_cache.py      # Redis caching
│   │   │   │   │   ├── memory_cache.py     # In-memory caching
│   │   │   │   │   ├── dmi_cache.py        # DMI momentum cache
│   │   │   │   │   └── cache_strategies.py # Caching strategies
│   │   │   │   ├── 📁 api/                 # API layer
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── routes.py           # FastAPI routes
│   │   │   │   │   ├── models.py           # Pydantic models
│   │   │   │   │   ├── dependencies.py     # FastAPI dependencies
│   │   │   │   │   └── middleware.py       # Custom middleware
│   │   │   │   ├── 📁 monitoring/          # Monitoring & metrics
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── metrics.py          # Prometheus metrics
│   │   │   │   │   ├── health_check.py     # Health endpoints
│   │   │   │   │   ├── logging_config.py   # Logging configuration
│   │   │   │   │   └── tracing.py          # Distributed tracing
│   │   │   │   ├── 📁 config/              # Configuration management
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── settings.py         # Application settings
│   │   │   │   │   ├── database.py         # Database configuration
│   │   │   │   │   └── security.py         # Security configuration
│   │   │   │   └── 📁 utils/               # Utility functions
│   │   │   │       ├── __init__.py
│   │   │   │       ├── validators.py       # Input validation
│   │   │   │       ├── exceptions.py       # Custom exceptions
│   │   │   │       └── helpers.py          # Helper functions
│   │   │   └── main.py                     # Application entry point
│   │   ├── 📁 tests/                       # Test suite
│   │   │   ├── 📁 unit/                    # Unit tests
│   │   │   │   ├── test_inference_engine.py
│   │   │   │   ├── test_mining.py
│   │   │   │   ├── test_cache.py
│   │   │   │   └── test_api.py
│   │   │   ├── 📁 integration/             # Integration tests
│   │   │   │   ├── test_end_to_end.py
│   │   │   │   ├── test_redis_integration.py
│   │   │   │   └── test_faiss_integration.py
│   │   │   ├── 📁 performance/             # Performance tests
│   │   │   │   ├── test_latency.py
│   │   │   │   ├── test_throughput.py
│   │   │   │   └── test_memory_usage.py
│   │   │   ├── 📁 fixtures/                # Test fixtures
│   │   │   │   ├── sample_models/
│   │   │   │   ├── test_data/
│   │   │   │   └── mock_responses/
│   │   │   └── conftest.py                 # Pytest configuration
│   │   ├── 📁 models/                      # ML models & artifacts
│   │   │   ├── 📁 quantized/               # Quantized models
│   │   │   │   ├── llama-7b-4bit.bin
│   │   │   │   └── llama-13b-4bit.bin
│   │   │   ├── 📁 embeddings/              # Embedding models
│   │   │   │   ├── sentence-transformers/
│   │   │   │   └── custom-embeddings/
│   │   │   └── 📁 configs/                 # Model configurations
│   │   │       ├── llama-7b-config.json
│   │   │       └── embedding-config.json
│   │   ├── 📁 k8s/                         # Kubernetes manifests
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── configmap.yaml
│   │   │   ├── pvc.yaml                    # Persistent volume claims
│   │   │   └── hpa.yaml                    # Horizontal Pod Autoscaler
│   │   ├── requirements.txt                # Python dependencies
│   │   ├── requirements-dev.txt            # Development dependencies
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml              # Local development
│   │   └── README.md
│   ├── 📁 vector-store/                    # Vector database service
│   │   ├── 📁 src/                         # Source code
│   │   │   └── 📁 vorta/vectorstore/       # Vector store implementation
│   │   │       ├── 📁 faiss/               # FAISS implementation
│   │   │       │   ├── __init__.py
│   │   │       │   ├── index_manager.py    # FAISS index management
│   │   │       │   ├── search_engine.py    # Search algorithms
│   │   │       │   └── persistence.py      # Index persistence
│   │   │       ├── 📁 weaviate/            # Weaviate integration
│   │   │       │   ├── __init__.py
│   │   │       │   ├── client.py           # Weaviate client
│   │   │       │   └── schema.py           # Schema management
│   │   │       ├── 📁 api/                 # API layer
│   │   │       │   ├── __init__.py
│   │   │       │   ├── routes.py           # FastAPI routes
│   │   │       │   └── models.py           # Pydantic models
│   │   │       └── 📁 utils/               # Utilities
│   │   │           ├── __init__.py
│   │   │           ├── vector_utils.py     # Vector operations
│   │   │           └── index_utils.py      # Index utilities
│   │   ├── 📁 tests/                       # Tests
│   │   ├── 📁 k8s/                         # Kubernetes manifests
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── 📁 orchestrator/                    # Cluster orchestration service
│   │   ├── 📁 src/                         # Source code
│   │   │   └── 📁 vorta/orchestrator/      # Orchestrator implementation
│   │   │       ├── 📁 scheduler/           # Task scheduling
│   │   │       │   ├── __init__.py
│   │   │       │   ├── load_balancer.py    # Load balancing
│   │   │       │   ├── task_queue.py       # Task queue management
│   │   │       │   └── priority_scheduler.py # Priority scheduling
│   │   │       ├── 📁 node_manager/        # Node management
│   │   │       │   ├── __init__.py
│   │   │       │   ├── node_registry.py    # Node registration
│   │   │       │   ├── health_monitor.py   # Node health monitoring
│   │   │       │   └── resource_tracker.py # Resource tracking
│   │   │       ├── 📁 communication/       # Inter-node communication
│   │   │       │   ├── __init__.py
│   │   │       │   ├── message_broker.py   # Message broker
│   │   │       │   ├── grpc_client.py      # gRPC client
│   │   │       │   └── event_publisher.py  # Event publishing
│   │   │       └── 📁 api/                 # API layer
│   │   │           ├── __init__.py
│   │   │           ├── routes.py           # FastAPI routes
│   │   │           └── models.py           # Pydantic models
│   │   ├── 📁 tests/                       # Tests
│   │   ├── 📁 k8s/                         # Kubernetes manifests
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── README.md
│   └── 📁 monitoring/                      # Monitoring & observability service
│       ├── 📁 prometheus/                  # Prometheus configuration
│       │   ├── prometheus.yml              # Prometheus config
│       │   ├── rules/                      # Alert rules
│       │   │   ├── vorta-alerts.yml
│       │   │   └── infrastructure-alerts.yml
│       │   └── targets/                    # Service discovery
│       │       └── targets.json
│       ├── 📁 grafana/                     # Grafana dashboards
│       │   ├── dashboards/                 # Dashboard definitions
│       │   │   ├── vorta-overview.json
│       │   │   ├── inference-performance.json
│       │   │   ├── cluster-health.json
│       │   │   └── business-metrics.json
│       │   ├── provisioning/               # Provisioning config
│       │   │   ├── dashboards/
│       │   │   └── datasources/
│       │   └── grafana.ini                 # Grafana configuration
│       ├── 📁 alertmanager/                # Alert manager
│       │   ├── alertmanager.yml            # Alert routing config
│       │   └── templates/                  # Alert templates
│       ├── 📁 jaeger/                      # Distributed tracing
│       │   └── jaeger-config.yml
│       └── 📁 elk/                         # Elasticsearch, Logstash, Kibana
│           ├── elasticsearch/
│           ├── logstash/
│           └── kibana/
├── 📁 infrastructure/                      # Infrastructure as Code
│   ├── 📁 terraform/                       # Terraform configurations
│   │   ├── 📁 environments/                # Environment-specific configs
│   │   │   ├── 📁 development/             # Development environment
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   └── terraform.tfvars
│   │   │   ├── 📁 staging/                 # Staging environment
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   └── terraform.tfvars
│   │   │   └── 📁 production/              # Production environment
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       ├── outputs.tf
│   │   │       └── terraform.tfvars
│   │   ├── 📁 modules/                     # Reusable Terraform modules
│   │   │   ├── 📁 vpc/                     # VPC module
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   ├── 📁 eks/                     # EKS cluster module
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   ├── 📁 rds/                     # RDS module
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   └── 📁 redis/                   # Redis module
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   └── 📁 shared/                      # Shared configurations
│   │       ├── backend.tf                  # Terraform backend
│   │       ├── providers.tf                # Provider configurations
│   │       └── versions.tf                 # Version constraints
│   ├── 📁 kubernetes/                      # Kubernetes manifests
│   │   ├── 📁 base/                        # Base configurations
│   │   │   ├── 📁 namespaces/              # Namespace definitions
│   │   │   │   ├── vorta-dev.yaml
│   │   │   │   ├── vorta-staging.yaml
│   │   │   │   └── vorta-prod.yaml
│   │   │   ├── 📁 rbac/                    # RBAC configurations
│   │   │   │   ├── service-accounts.yaml
│   │   │   │   ├── roles.yaml
│   │   │   │   └── role-bindings.yaml
│   │   │   ├── 📁 network-policies/        # Network policies
│   │   │   │   ├── default-deny.yaml
│   │   │   │   └── service-mesh.yaml
│   │   │   └── 📁 secrets/                 # Secret templates
│   │   │       ├── tls-certificates.yaml
│   │   │       └── api-keys.yaml
│   │   ├── 📁 overlays/                    # Kustomize overlays
│   │   │   ├── 📁 development/             # Development overlay
│   │   │   │   ├── kustomization.yaml
│   │   │   │   └── patches/
│   │   │   ├── 📁 staging/                 # Staging overlay
│   │   │   │   ├── kustomization.yaml
│   │   │   │   └── patches/
│   │   │   └── 📁 production/              # Production overlay
│   │   │       ├── kustomization.yaml
│   │   │       └── patches/
│   │   └── 📁 helm/                        # Helm charts
│   │       ├── 📁 vorta-platform/          # Main platform chart
│   │       │   ├── Chart.yaml
│   │       │   ├── values.yaml
│   │       │   ├── values-dev.yaml
│   │       │   ├── values-staging.yaml
│   │       │   ├── values-prod.yaml
│   │       │   └── 📁 templates/           # Helm templates
│   │       │       ├── deployment.yaml
│   │       │       ├── service.yaml
│   │       │       ├── ingress.yaml
│   │       │       └── configmap.yaml
│   │       └── 📁 dependencies/            # Third-party charts
│   │           ├── prometheus/
│   │           ├── grafana/
│   │           └── redis/
│   ├── 📁 ansible/                         # Ansible playbooks
│   │   ├── 📁 playbooks/                   # Ansible playbooks
│   │   │   ├── site.yml                    # Main playbook
│   │   │   ├── monitoring.yml              # Monitoring setup
│   │   │   └── security.yml                # Security hardening
│   │   ├── 📁 roles/                       # Ansible roles
│   │   │   ├── 📁 common/                  # Common role
│   │   │   ├── 📁 docker/                  # Docker installation
│   │   │   ├── 📁 kubernetes/              # Kubernetes setup
│   │   │   └── 📁 monitoring/              # Monitoring setup
│   │   ├── 📁 inventory/                   # Inventory files
│   │   │   ├── development/
│   │   │   ├── staging/
│   │   │   └── production/
│   │   └── ansible.cfg                     # Ansible configuration
│   └── 📁 docker/                          # Docker configurations
│       ├── 📁 base/                        # Base images
│       │   ├── Dockerfile.python           # Python base image
│       │   ├── Dockerfile.java             # Java base image
│       │   └── Dockerfile.alpine           # Alpine base image
│       └── docker-compose.yml              # Local development stack
├── 📁 shared/                              # Shared libraries & utilities
│   ├── 📁 libraries/                       # Shared code libraries
│   │   ├── 📁 python/                      # Python shared library
│   │   │   ├── 📁 vorta_common/            # Common Python package
│   │   │   │   ├── __init__.py
│   │   │   │   ├── 📁 auth/                # Authentication utilities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── jwt_handler.py      # JWT handling
│   │   │   │   │   └── api_key_validator.py # API key validation
│   │   │   │   ├── 📁 monitoring/          # Monitoring utilities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── metrics.py          # Metrics collection
│   │   │   │   │   └── logging.py          # Logging utilities
│   │   │   │   ├── 📁 database/            # Database utilities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── connection.py       # DB connections
│   │   │   │   │   └── migrations.py       # Migration utilities
│   │   │   │   └── 📁 utils/               # General utilities
│   │   │   │       ├── __init__.py
│   │   │   │       ├── config.py           # Configuration handling
│   │   │   │       ├── validators.py       # Input validation
│   │   │   │       └── exceptions.py       # Common exceptions
│   │   │   ├── setup.py                    # Package setup
│   │   │   ├── requirements.txt            # Dependencies
│   │   │   └── README.md                   # Library documentation
│   │   └── 📁 java/                        # Java shared library
│   │       ├── 📁 src/main/java/com/vorta/common/ # Java source
│   │       │   ├── 📁 auth/                # Authentication
│   │       │   ├── 📁 monitoring/          # Monitoring
│   │       │   ├── 📁 database/            # Database
│   │       │   └── 📁 utils/               # Utilities
│   │       ├── pom.xml                     # Maven configuration
│   │       └── README.md                   # Library documentation
│   ├── 📁 configurations/                  # Shared configurations
│   │   ├── 📁 logging/                     # Logging configurations
│   │   │   ├── logback-spring.xml          # Logback config
│   │   │   ├── log4j2.xml                  # Log4j config
│   │   │   └── python-logging.yaml         # Python logging config
│   │   ├── 📁 monitoring/                  # Monitoring configs
│   │   │   ├── prometheus-rules.yaml       # Prometheus rules
│   │   │   ├── grafana-dashboards.json     # Grafana dashboards
│   │   │   └── jaeger-config.yaml          # Jaeger config
│   │   └── 📁 security/                    # Security configurations
│   │       ├── jwt-config.yaml             # JWT configuration
│   │       ├── tls-config.yaml             # TLS configuration
│   │       └── rbac-policies.yaml          # RBAC policies
│   └── 📁 schemas/                         # Shared data schemas
│       ├── 📁 api/                         # API schemas
│       │   ├── inference-request.json      # Inference request schema
│       │   ├── inference-response.json     # Inference response schema
│       │   └── error-response.json         # Error response schema
│       ├── 📁 events/                      # Event schemas
│       │   ├── inference-started.json      # Inference started event
│       │   ├── inference-completed.json    # Inference completed event
│       │   └── node-status-changed.json    # Node status event
│       └── 📁 database/                    # Database schemas
│           ├── users.sql                   # User table schema
│           ├── api_keys.sql                # API keys table schema
│           └── audit_log.sql               # Audit log schema
├── 📁 scripts/                             # Build & deployment scripts
│   ├── 📁 build/                           # Build scripts
│   │   ├── build-all.sh                    # Build all services
│   │   ├── build-service.sh                # Build single service
│   │   ├── package.sh                      # Package artifacts
│   │   └── version.sh                      # Version management
│   ├── 📁 deploy/                          # Deployment scripts
│   │   ├── deploy-dev.sh                   # Deploy to development
│   │   ├── deploy-staging.sh               # Deploy to staging
│   │   ├── deploy-prod.sh                  # Deploy to production
│   │   ├── rollback.sh                     # Rollback deployment
│   │   └── blue-green-deploy.sh            # Blue-green deployment
│   ├── 📁 maintenance/                     # Maintenance scripts
│   │   ├── backup.sh                       # Backup script
│   │   ├── restore.sh                      # Restore script
│   │   ├── cleanup.sh                      # Cleanup old resources
│   │   └── health-check.sh                 # Health check script
│   ├── 📁 development/                     # Development scripts
│   │   ├── setup-dev-env.sh                # Setup development environment
│   │   ├── run-tests.sh                    # Run test suite
│   │   ├── lint.sh                         # Code linting
│   │   └── format.sh                       # Code formatting
│   └── 📁 security/                        # Security scripts
│       ├── security-scan.sh                # Security scanning
│       ├── vulnerability-check.sh          # Vulnerability checking
│       └── generate-certificates.sh        # Certificate generation
├── 📁 tests/                               # Integration & E2E tests
│   ├── 📁 integration/                     # Cross-service integration tests
│   │   ├── 📁 api/                         # API integration tests
│   │   │   ├── test_inference_flow.py      # End-to-end inference test
│   │   │   ├── test_authentication.py      # Auth integration test
│   │   │   └── test_error_handling.py      # Error handling test
│   │   ├── 📁 performance/                 # Performance tests
│   │   │   ├── test_load.py                # Load testing
│   │   │   ├── test_stress.py              # Stress testing
│   │   │   └── test_endurance.py           # Endurance testing
│   │   └── 📁 security/                    # Security tests
│   │       ├── test_authentication.py      # Auth security tests
│   │       ├── test_authorization.py       # Authorization tests
│   │       └── test_data_protection.py     # Data protection tests
│   ├── 📁 e2e/                             # End-to-end tests
│   │   ├── 📁 playwright/                  # Browser-based tests
│   │   │   ├── tests/                      # Test files
│   │   │   ├── page-objects/               # Page object models
│   │   │   └── fixtures/                   # Test fixtures
│   │   └── 📁 api/                         # API end-to-end tests
│   │       ├── test_user_journey.py        # User journey tests
│   │       ├── test_business_flows.py      # Business flow tests
│   │       └── test_edge_cases.py          # Edge case tests
│   ├── 📁 chaos/                           # Chaos engineering tests
│   │   ├── test_node_failure.py            # Node failure scenarios
│   │   ├── test_network_partition.py       # Network partition tests
│   │   └── test_resource_exhaustion.py     # Resource exhaustion tests
│   ├── 📁 fixtures/                        # Test fixtures
│   │   ├── 📁 data/                        # Test data
│   │   │   ├── sample_requests.json        # Sample API requests
│   │   │   ├── test_models/                # Test ML models
│   │   │   └── mock_responses/             # Mock API responses
│   │   ├── 📁 environments/                # Test environments
│   │   │   ├── docker-compose.test.yml     # Test environment setup
│   │   │   └── k8s-test-env.yaml           # K8s test environment
│   │   └── 📁 certificates/                # Test certificates
│   │       ├── test-ca.crt                 # Test CA certificate
│   │       └── test-server.crt             # Test server certificate
│   ├── pytest.ini                         # Pytest configuration
│   ├── conftest.py                         # Global test configuration
│   └── requirements-test.txt               # Test dependencies
├── 📁 tools/                               # Development & operational tools
│   ├── 📁 cli/                             # Command-line tools
│   │   ├── 📁 vorta-cli/                   # Main CLI tool
│   │   │   ├── 📁 src/                     # CLI source code
│   │   │   │   ├── 📁 commands/            # CLI commands
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── deploy.py           # Deployment commands
│   │   │   │   │   ├── monitor.py          # Monitoring commands
│   │   │   │   │   ├── config.py           # Configuration commands
│   │   │   │   │   └── debug.py            # Debugging commands
│   │   │   │   ├── 📁 utils/               # CLI utilities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── config_loader.py    # Configuration loading
│   │   │   │   │   ├── output_formatter.py # Output formatting
│   │   │   │   │   └── api_client.py       # API client
│   │   │   │   └── main.py                 # CLI entry point
│   │   │   ├── requirements.txt            # CLI dependencies
│   │   │   ├── setup.py                    # CLI package setup
│   │   │   └── README.md                   # CLI documentation
│   │   └── 📁 scripts/                     # Utility scripts
│   │       ├── cluster-info.sh             # Cluster information script
│   │       ├── log-collector.sh            # Log collection script
│   │       └── performance-report.sh       # Performance reporting
│   ├── 📁 monitoring/                      # Monitoring tools
│   │   ├── 📁 dashboards/                  # Custom dashboards
│   │   │   ├── vorta-business-metrics.json # Business metrics dashboard
│   │   │   ├── vorta-technical-health.json # Technical health dashboard
│   │   │   └── vorta-cost-analysis.json    # Cost analysis dashboard
│   │   ├── 📁 alerts/                      # Alert configurations
│   │   │   ├── sla-alerts.yaml             # SLA violation alerts
│   │   │   ├── performance-alerts.yaml     # Performance alerts
│   │   │   └── security-alerts.yaml        # Security alerts
│   │   └── 📁 exporters/                   # Custom metric exporters
│   │       ├── vorta-metrics-exporter.py   # VORTA-specific metrics
│   │       └── business-metrics-exporter.py # Business metrics
│   ├── 📁 development/                     # Development tools
│   │   ├── 📁 generators/                  # Code generators
│   │   │   ├── service-generator.py        # Service scaffold generator
│   │   │   ├── api-generator.py            # API code generator
│   │   │   └── test-generator.py           # Test code generator
│   │   ├── 📁 validators/                  # Validation tools
│   │   │   ├── config-validator.py         # Configuration validator
│   │   │   ├── api-validator.py            # API contract validator
│   │   │   └── schema-validator.py         # Schema validator
│   │   └── 📁 profilers/                   # Performance profilers
│   │       ├── memory-profiler.py          # Memory usage profiler
│   │       ├── cpu-profiler.py             # CPU usage profiler
│   │       └── io-profiler.py              # I/O profiler
│   └── 📁 security/                        # Security tools
│       ├── 📁 scanners/                    # Security scanners
│       │   ├── dependency-scanner.py       # Dependency vulnerability scanner
│       │   ├── container-scanner.py        # Container security scanner
│       │   └── api-security-scanner.py     # API security scanner
│       ├── 📁 generators/                  # Security generators
│       │   ├── certificate-generator.py    # Certificate generation
│       │   ├── key-generator.py            # Key generation
│       │   └── token-generator.py          # Token generation
│       └── 📁 auditors/                    # Security auditors
│           ├── access-auditor.py           # Access pattern auditor
│           ├── compliance-auditor.py       # Compliance checker
│           └── vulnerability-auditor.py    # Vulnerability auditor
├── 📁 sdk/                                 # Software Development Kits
│   ├── 📁 python/                          # Python SDK
│   │   ├── 📁 vorta_sdk/                   # Main SDK package
│   │   │   ├── __init__.py
│   │   │   ├── 📁 client/                  # API client
│   │   │   │   ├── __init__.py
│   │   │   │   ├── http_client.py          # HTTP client
│   │   │   │   ├── auth_client.py          # Authentication client
│   │   │   │   └── inference_client.py     # Inference client
│   │   │   ├── 📁 models/                  # Data models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── requests.py             # Request models
│   │   │   │   ├── responses.py            # Response models
│   │   │   │   └── errors.py               # Error models
│   │   │   ├── 📁 utils/                   # SDK utilities
│   │   │   │   ├── __init__.py
│   │   │   │   ├── config.py               # Configuration utilities
│   │   │   │   ├── retry.py                # Retry mechanisms
│   │   │   │   └── logging.py              # Logging utilities
│   │   │   └── 📁 examples/                # Usage examples
│   │   │       ├── __init__.py
│   │   │       ├── basic_inference.py      # Basic inference example
│   │   │       ├── batch_processing.py     # Batch processing example
│   │   │       └── async_inference.py      # Async inference example
│   │   ├── 📁 tests/                       # SDK tests
│   │   │   ├── test_client.py              # Client tests
│   │   │   ├── test_models.py              # Model tests
│   │   │   └── test_integration.py         # Integration tests
│   │   ├── 📁 docs/                        # SDK documentation
│   │   │   ├── quickstart.md               # Quick start guide
│   │   │   ├── api-reference.md            # API reference
│   │   │   └── examples.md                 # Examples documentation
│   │   ├── setup.py                        # Package setup
│   │   ├── requirements.txt                # Dependencies
│   │   ├── README.md                       # SDK documentation
│   │   └── CHANGELOG.md                    # Version changelog
│   ├── 📁 javascript/                      # JavaScript/TypeScript SDK
│   │   ├── 📁 src/                         # Source code
│   │   │   ├── 📁 client/                  # API client
│   │   │   │   ├── http-client.ts          # HTTP client
│   │   │   │   ├── auth-client.ts          # Authentication client
│   │   │   │   └── inference-client.ts     # Inference client
│   │   │   ├── 📁 types/                   # TypeScript types
│   │   │   │   ├── requests.ts             # Request types
│   │   │   │   ├── responses.ts            # Response types
│   │   │   │   └── errors.ts               # Error types
│   │   │   ├── 📁 utils/                   # Utilities
│   │   │   │   ├── config.ts               # Configuration
│   │   │   │   ├── retry.ts                # Retry logic
│   │   │   │   └── logger.ts               # Logging
│   │   │   └── index.ts                    # Main export
│   │   ├── 📁 tests/                       # Tests
│   │   │   ├── client.test.ts              # Client tests
│   │   │   ├── types.test.ts               # Type tests
│   │   │   └── integration.test.ts         # Integration tests
│   │   ├── 📁 examples/                    # Usage examples
│   │   │   ├── basic-usage.js              # Basic usage example
│   │   │   ├── node-example.js             # Node.js example
│   │   │   └── browser-example.html        # Browser example
│   │   ├── package.json                    # NPM package config
│   │   ├── tsconfig.json                   # TypeScript config
│   │   ├── webpack.config.js               # Webpack config
│   │   ├── README.md                       # SDK documentation
│   │   └── CHANGELOG.md                    # Version changelog
│   └── 📁 java/                            # Java SDK
│       ├── 📁 src/                         # Source code
│       │   ├── 📁 main/java/com/vorta/sdk/ # Main source
│       │   │   ├── 📁 client/              # API client
│       │   │   │   ├── VortaClient.java    # Main client
│       │   │   │   ├── AuthClient.java     # Auth client
│       │   │   │   └── InferenceClient.java # Inference client
│       │   │   ├── 📁 model/               # Data models
│       │   │   │   ├── request/            # Request models
│       │   │   │   ├── response/           # Response models
│       │   │   │   └── error/              # Error models
│       │   │   ├── 📁 util/                # Utilities
│       │   │   │   ├── ConfigUtil.java     # Configuration utility
│       │   │   │   ├── RetryUtil.java      # Retry utility
│       │   │   │   └── LoggingUtil.java    # Logging utility
│       │   │   └── 📁 exception/           # Custom exceptions
│       │   │       ├── VortaException.java # Base exception
│       │   │       └── ApiException.java   # API exception
│       │   └── 📁 test/java/com/vorta/sdk/ # Test source
│       │       ├── client/                 # Client tests
│       │       ├── model/                  # Model tests
│       │       └── integration/            # Integration tests
│       ├── 📁 examples/                    # Usage examples
│       │   ├── BasicInference.java         # Basic inference example
│       │   ├── BatchProcessing.java        # Batch processing example
│       │   └── AsyncInference.java         # Async inference example
│       ├── pom.xml                         # Maven configuration
│       ├── README.md                       # SDK documentation
│       └── CHANGELOG.md                    # Version changelog
├── 📁 config/                              # Global configuration files
│   ├── 📁 environments/                    # Environment configurations
│   │   ├── development.env                 # Development environment
│   │   ├── staging.env                     # Staging environment
│   │   ├── production.env                  # Production environment
│   │   └── local.env                       # Local development
│   ├── 📁 secrets/                         # Secret management (encrypted)
│   │   ├── development-secrets.yaml        # Development secrets
│   │   ├── staging-secrets.yaml            # Staging secrets
│   │   └── production-secrets.yaml         # Production secrets
│   ├── 📁 feature-flags/                   # Feature flag configurations
│   │   ├── development-flags.yaml          # Development flags
│   │   ├── staging-flags.yaml              # Staging flags
│   │   └── production-flags.yaml           # Production flags
│   └── 📁 policies/                        # Policy configurations
│       ├── security-policies.yaml          # Security policies
│       ├── network-policies.yaml           # Network policies
│       └── resource-policies.yaml          # Resource policies
├── 📁 storage/                             # Persistent data (development only)
│   ├── 📁 databases/                       # Database data
│   │   ├── postgresql/                     # PostgreSQL data
│   │   └── redis/                          # Redis data
│   ├── 📁 logs/                            # Application logs
│   │   ├── application/                    # Application logs
│   │   ├── audit/                          # Audit logs
│   │   └── access/                         # Access logs
│   ├── 📁 models/                          # ML model storage
│   │   ├── trained/                        # Trained models
│   │   ├── quantized/                      # Quantized models
│   │   └── embeddings/                     # Embedding models
│   └── 📁 backups/                         # Backup storage
│       ├── database/                       # Database backups
│       └── configuration/                  # Configuration backups
├── 📁 .devcontainer/                       # Development container
│   ├── devcontainer.json                   # Dev container configuration
│   ├── Dockerfile                          # Dev container dockerfile
│   └── docker-compose.yml                  # Dev container services
├── 📁 .vscode/                             # VS Code configuration
│   ├── settings.json                       # VS Code settings
│   ├── launch.json                         # Debug configurations
│   ├── tasks.json                          # VS Code tasks
│   └── extensions.json                     # Recommended extensions
├── .gitignore                              # Git ignore rules
├── .gitattributes                          # Git attributes
├── .pre-commit-config.yaml                 # Pre-commit hooks
├── .editorconfig                           # Editor configuration
├── docker-compose.yml                      # Main docker compose
├── docker-compose.override.yml             # Local overrides
├── docker-compose.prod.yml                 # Production compose
├── Makefile                                # Build automation
├── README.md                               # Main project documentation
├── CONTRIBUTING.md                         # Contribution guidelines
├── CODE_OF_CONDUCT.md                      # Code of conduct
├── LICENSE                                 # Project license
├── CHANGELOG.md                            # Project changelog
├── SECURITY.md                             # Security policy
└── VERSION                                 # Version file
"""

# Content voor belangrijke bestanden
FILE_CONTENTS = {
    "vorta-platform/.gitignore": """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# PEP 582; used by PDM, PEP 582 proposal
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype
.pytype/

# Cython debug symbols
cython_debug/
""",
    "vorta-platform/README.md": """
# VORTA™ Platform

Welcome to the VORTA™ Platform, an enterprise-grade, production-ready AI infrastructure designed for radical efficiency and performance.

## 📋 Overview

This platform is built on a microservices architecture, featuring:
- **Inference Engine**: The core AI/ML service for high-performance inference.
- **Vector Store**: A specialized service for managing and searching high-dimensional vectors.
- **API Gateway**: The single entry point for all client requests.
- **Orchestrator**: Manages the cluster, nodes, and tasks.

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Kubernetes (minikube for local development)
- Python 3.11+
- Terraform

### Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/vorta-platform.git
    cd vorta-platform
    ```

2.  **Setup the development environment:**
    This will create a virtual environment and install all necessary dependencies.
    ```bash
    make setup-dev
    ```

3.  **Start the local development stack:**
    This will spin up all required services using Docker Compose.
    ```bash
    make start-dev
    ```

4.  **Run the tests:**
    ```bash
    make test
    ```

## 🏗️ Project Structure

For a detailed overview of the project structure, please see `VORTA_Project_Structure.md`.

## 🤝 Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the [LICENSE](LICENSE) file - see the details.
""",
    "vorta-platform/services/inference-engine/requirements.txt": """
fastapi
uvicorn[standard]
pydantic
torch
transformers
faiss-cpu
redis
prometheus-client
python-json-logger
""",
    "vorta-platform/services/inference-engine/requirements-dev.txt": """
pytest
pytest-asyncio
httpx
black
isort
mypy
""",
    "vorta-platform/services/inference-engine/Dockerfile": """
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src/vorta /app/vorta
COPY ./src/main.py /app/main.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "vorta-platform/docker-compose.yml": """
version: '3.8'

services:
  redis:
    image: "redis:alpine"
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: "postgres:15-alpine"
    environment:
      POSTGRES_DB: vorta
      POSTGRES_USER: vorta
      POSTGRES_PASSWORD: vorta_dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  inference-engine:
    build:
      context: ./services/inference-engine
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis

volumes:
  redis_data:
  postgres_data:
""",
    "vorta-platform/Makefile": """
.PHONY: help setup-dev start-dev stop-dev test lint format clean

help:
	@echo "VORTA Development Commands:"
	@echo "  setup-dev    - Setup development environment"
	@echo "  start-dev    - Start local development services via Docker Compose"
	@echo "  stop-dev     - Stop local development services"
	@echo "  test         - Run tests for all services"
	@echo "  lint         - Run linting and type checking"
	@echo "  format       - Format code using black and isort"
	@echo "  clean        - Clean up temporary files"

setup-dev:
	@echo "🚀 Setting up VORTA development environment..."
	python -m venv venv
	@echo "Virtual environment created."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r services/inference-engine/requirements.txt -r services/inference-engine/requirements-dev.txt
	@echo "✅ Development environment ready!"

start-dev:
	@echo "🐳 Starting development services..."
	docker-compose up -d
	@echo "✅ Services started!"

stop-dev:
	@echo "🛑 Stopping development services..."
	docker-compose down

test:
	@echo "🧪 Running tests..."
	./venv/bin/pytest services/

lint:
	@echo "🔍 Running linting..."
	./venv/bin/mypy services/
	./venv/bin/black --check services/
	./venv/bin/isort --check-only services/

format:
	@echo "🎨 Formatting code..."
	./venv/bin/black services/
	./venv/bin/isort services/

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
"""
}

def create_file_with_content(path, content_key):
    """Maakt een bestand en schrijft er de standaardinhoud naartoe."""
    try:
        content = FILE_CONTENTS.get(content_key, f"# Placeholder for {os.path.basename(path)}\n")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"   📄 Created file: {path}")
    except IOError as e:
        print(f"   ❌ Error creating file {path}: {e}")

def main():
    """Parse de structuur en maak de mappen en bestanden aan."""
    lines = PROJECT_STRUCTURE.strip().split('\n')
    path_stack = []
    
    print("🚀 Starting VORTA project structure creation...")
    
    for line in lines:
        # Negeer lege regels en regels die alleen commentaar bevatten
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Parse de tree structuur karakters om diepte te bepalen
        # Vervang de verschillende tree karakters door een standaard marker
        normalized_line = line.replace('├──', '→').replace('└──', '→').replace('│   ', '│').replace('    ', '│')
        
        # Tel de diepte door │ en spaties te tellen voor de naam
        depth = 0
        i = 0
        while i < len(normalized_line) and (normalized_line[i] in '│ ├└─'):
            if normalized_line[i] == '│':
                depth += 1
            i += 1
        
        # Extract de naam door alle tree karakters en emojis te verwijderen
        name_part = line.split('#')[0]  # Verwijder commentaar
        name_part = re.sub(r'^[│├└─\s]*', '', name_part)  # Verwijder tree karakters
        name_part = re.sub(r'📁\s*', '', name_part)  # Verwijder folder emoji
        name_part = name_part.strip().rstrip('/')  # Verwijder trailing slash en whitespace
        
        if not name_part:
            continue
            
        # Bepaal of het een directory is
        is_directory = ('📁' in line or 
                       line.rstrip().endswith('/') or 
                       '.' not in name_part.split('/')[-1] or
                       name_part in ['workflows', 'ISSUE_TEMPLATE', 'architecture', 'adr', 'api', 'deployment', 'development', 'user-guides'])
        
        # Adjust path stack based on depth
        while len(path_stack) > depth:
            path_stack.pop()
            
        # Add current item to path stack
        path_stack.append(name_part)
        current_path = os.path.join(*path_stack)
        
        try:
            if is_directory:
                os.makedirs(current_path, exist_ok=True)
                print(f"📁 Created directory: {current_path}")
            else:
                # Ensure parent directory exists
                parent_dir = os.path.dirname(current_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Create file with content
                content_key = "/".join(path_stack)
                create_file_with_content(current_path, content_key)
                
        except Exception as e:
            print(f"❌ Error processing {current_path}: {e}")
            continue

    print("\n✅ VORTA project structure created successfully!")
    print("👉 To get started, run 'make setup-dev' and then 'make start-dev'.")

if __name__ == "__main__":
    main()
