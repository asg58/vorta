#!/usr/bin/env python3
"""
VORTA GitHub Repository Setup Script
Automates the setup of GitHub repository settings, branch protection, and secrets
"""

import json
import subprocess
import sys


def run_command(command, check=True, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e.stderr if e.stderr else str(e)}")
        return None

def setup_github_repository():
    """Setup GitHub repository with all necessary configurations"""
    print("🚀 Setting up VORTA GitHub Repository...")
    
    # Check if gh CLI is installed
    gh_version = run_command("gh --version")
    if not gh_version:
        print("❌ GitHub CLI (gh) is not installed. Please install it first.")
        print("   Visit: https://cli.github.com/")
        return False
    
    print(f"✅ GitHub CLI found: {gh_version.split()[2]}")
    
    # Check if user is authenticated
    auth_status = run_command("gh auth status", check=False)
    if not auth_status or "Logged in" not in auth_status:
        print("❌ Not authenticated with GitHub. Please run: gh auth login")
        return False
    
    print("✅ GitHub authentication verified")
    
    # Get current repository info
    repo_info = run_command("gh repo view --json name,owner", check=False)
    if not repo_info:
        print("❌ Not in a GitHub repository. Please run from the repository root.")
        return False
    
    repo_data = json.loads(repo_info)
    repo_name = repo_data['name']
    owner = repo_data['owner']['login']
    full_repo = f"{owner}/{repo_name}"
    
    print(f"📦 Repository: {full_repo}")
    
    # Setup repository settings
    setup_repository_settings(full_repo)
    
    # Setup branch protection
    setup_branch_protection(full_repo)
    
    # Setup repository secrets
    setup_repository_secrets(full_repo)
    
    # Setup environments
    setup_environments(full_repo)
    
    # Setup labels
    setup_labels(full_repo)
    
    print("🎉 GitHub repository setup completed successfully!")
    return True

def setup_repository_settings(repo):
    """Configure repository settings"""
    print("\n⚙️ Configuring repository settings...")
    
    settings = {
        "has_issues": True,
        "has_projects": True,
        "has_wiki": False,
        "has_downloads": True,
        "default_branch": "main",
        "allow_squash_merge": True,
        "allow_merge_commit": False,
        "allow_rebase_merge": True,
        "delete_branch_on_merge": True,
        "vulnerability_alerts": True,
        "automated_security_fixes": True
    }
    
    for setting, value in settings.items():
        command = f'gh api repos/{repo} --method PATCH --field {setting}={str(value).lower()}'
        result = run_command(command, check=False)
        if result is not None:
            print(f"  ✅ {setting}: {value}")
        else:
            print(f"  ⚠️ Failed to set {setting}")

def setup_branch_protection(repo):
    """Setup branch protection rules"""
    print("\n🛡️ Setting up branch protection...")
    
    protection_rules = {
        "required_status_checks": {
            "strict": True,
            "contexts": [
                "🔄 Continuous Integration / code-quality",
                "🔄 Continuous Integration / test-python-services",
                "🔄 Continuous Integration / test-java-services",
                "🔄 Continuous Integration / test-sdks",
                "🔄 Continuous Integration / docker-builds",
                "🔄 Continuous Integration / integration-tests"
            ]
        },
        "enforce_admins": False,
        "required_pull_request_reviews": {
            "required_approving_review_count": 2,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
            "require_last_push_approval": True
        },
        "restrictions": None,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True
    }
    
    # Apply to main branch
    command = f'''gh api repos/{repo}/branches/main/protection \\
        --method PUT \\
        --raw-field required_status_checks='{json.dumps(protection_rules["required_status_checks"])}' \\
        --field enforce_admins={str(protection_rules["enforce_admins"]).lower()} \\
        --raw-field required_pull_request_reviews='{json.dumps(protection_rules["required_pull_request_reviews"])}' \\
        --field restrictions=null \\
        --field allow_force_pushes={str(protection_rules["allow_force_pushes"]).lower()} \\
        --field allow_deletions={str(protection_rules["allow_deletions"]).lower()} \\
        --field block_creations={str(protection_rules["block_creations"]).lower()} \\
        --field required_conversation_resolution={str(protection_rules["required_conversation_resolution"]).lower()}'''
    
    result = run_command(command, check=False)
    if result is not None:
        print("  ✅ Main branch protection enabled")
    else:
        print("  ⚠️ Failed to setup main branch protection")
    
    # Apply to develop branch (if it exists)
    develop_exists = run_command(f"gh api repos/{repo}/branches/develop", check=False)
    if develop_exists:
        # Lighter protection for develop branch
        develop_protection = protection_rules.copy()
        develop_protection["required_pull_request_reviews"]["required_approving_review_count"] = 1
        develop_protection["enforce_admins"] = False
        
        command = f'''gh api repos/{repo}/branches/develop/protection \\
            --method PUT \\
            --raw-field required_status_checks='{json.dumps(develop_protection["required_status_checks"])}' \\
            --field enforce_admins=false \\
            --raw-field required_pull_request_reviews='{json.dumps(develop_protection["required_pull_request_reviews"])}' \\
            --field restrictions=null \\
            --field allow_force_pushes=false \\
            --field allow_deletions=false'''
        
        result = run_command(command, check=False)
        if result is not None:
            print("  ✅ Develop branch protection enabled")
        else:
            print("  ⚠️ Failed to setup develop branch protection")

def setup_repository_secrets():
    """Setup repository secrets templates"""
    print("\n🔐 Repository secrets need manual setup...")
    print("  Please add the following secrets in GitHub Settings > Secrets and Variables > Actions:")
    
    secrets = {
        # Database secrets
        "STAGING_DATABASE_URL": "postgresql://user:pass@staging-db:5432/vorta",
        "PRODUCTION_DATABASE_URL": "postgresql://user:pass@prod-db:5432/vorta",
        
        # Redis secrets
        "STAGING_REDIS_URL": "redis://staging-redis:6379",
        "PRODUCTION_REDIS_URL": "redis://prod-redis:6379",
        
        # API keys
        "STAGING_API_KEY": "your-staging-api-key",
        "PRODUCTION_API_KEY": "your-production-api-key",
        
        # JWT secrets
        "STAGING_JWT_SECRET": "your-staging-jwt-secret",
        "PRODUCTION_JWT_SECRET": "your-production-jwt-secret",
        
        # Kubernetes configurations (base64 encoded)
        "KUBE_CONFIG_STAGING": "base64-encoded-staging-kubeconfig",
        "KUBE_CONFIG_PRODUCTION": "base64-encoded-production-kubeconfig",
        
        # Container registry
        "GHCR_TOKEN": "github-personal-access-token-with-packages-write"
    }
    
    for secret_name, description in secrets.items():
        print(f"    📝 {secret_name}: {description}")
    
    print("\n  💡 Tip: Use 'gh secret set SECRET_NAME' to add secrets via CLI")

def setup_environments(repo):
    """Setup GitHub environments"""
    print("\n🌍 Setting up environments...")
    
    environments = {
        "development": {
            "protection_rules": [],
            "deployment_branch_policy": {"protected_branches": False, "custom_branches": True}
        },
        "staging": {
            "protection_rules": [],
            "deployment_branch_policy": {"protected_branches": True, "custom_branches": False}
        },
        "production": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": []}],
            "deployment_branch_policy": {"protected_branches": True, "custom_branches": False}
        },
        "production-approval": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": []}],
            "deployment_branch_policy": {"protected_branches": True, "custom_branches": False}
        },
        "production-traffic-switch": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": []}],
            "deployment_branch_policy": {"protected_branches": True, "custom_branches": False}
        }
    }
    
    for env_name, config in environments.items():
        command = f'''gh api repos/{repo}/environments/{env_name} \\
            --method PUT \\
            --raw-field protection_rules='{json.dumps(config["protection_rules"])}' \\
            --raw-field deployment_branch_policy='{json.dumps(config["deployment_branch_policy"])}'\\'''
        
        result = run_command(command, check=False)
        if result is not None:
            print(f"  ✅ Environment '{env_name}' configured")
        else:
            print(f"  ⚠️ Failed to configure environment '{env_name}'")

def setup_labels(repo):
    """Setup repository labels"""
    print("\n🏷️ Setting up labels...")
    
    labels = [
        {"name": "security", "color": "d73a4a", "description": "Security related issues"},
        {"name": "vulnerability", "color": "d73a4a", "description": "Security vulnerability"},
        {"name": "dependencies", "color": "0366d6", "description": "Dependency related"},
        {"name": "automated", "color": "1d76db", "description": "Automated PR or issue"},
        {"name": "performance", "color": "f9d0c4", "description": "Performance related"},
        {"name": "ci/cd", "color": "0e8a16", "description": "CI/CD pipeline related"},
        {"name": "infrastructure", "color": "5319e7", "description": "Infrastructure related"},
        {"name": "urgent", "color": "b60205", "description": "Requires immediate attention"},
        {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
        {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
        {"name": "documentation", "color": "0075ca", "description": "Improvements or additions to documentation"},
        {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
        {"name": "help wanted", "color": "008672", "description": "Extra attention is needed"}
    ]
    
    for label in labels:
        command = f'''gh api repos/{repo}/labels \\
            --method POST \\
            --field name='{label["name"]}' \\
            --field color='{label["color"]}' \\
            --field description='{label["description"]}'\\'''
        
        result = run_command(command, check=False)
        if result is not None:
            print(f"  ✅ Label '{label['name']}' created")
        else:
            print(f"  ⚠️ Label '{label['name']}' may already exist")

def create_initial_issues():
    """Create initial project issues"""
    print("\n📋 Creating initial project issues...")
    
    issues = [
        {
            "title": "🔧 Setup Development Environment",
            "body": """## Development Environment Setup

### Tasks:
- [ ] Clone repository
- [ ] Install Python 3.11+
- [ ] Install Java 11+
- [ ] Install Node.js 18+
- [ ] Setup Docker and Docker Compose
- [ ] Install Kubernetes tools (kubectl, helm)
- [ ] Configure VS Code extensions
- [ ] Run initial tests

### Documentation:
- Review VORTA_Development_Environment.md
- Follow setup instructions in README.md

### Acceptance Criteria:
- All services can be built locally
- Tests pass in development environment
- VS Code debugging works
            """,
            "labels": ["good first issue", "documentation"]
        },
        {
            "title": "🏗️ Implement Core Inference Engine",
            "body": """## Core Inference Engine Implementation

### Scope:
Implementation of the VORTA inference engine service with optimized GPU acceleration.

### Key Components:
- [ ] FastAPI service structure
- [ ] GPU memory management
- [ ] Model loading and caching
- [ ] Batch processing optimization
- [ ] Performance monitoring

### Technical Requirements:
- Support for transformer models
- CUDA/ROCm acceleration
- Memory-efficient batching
- Comprehensive error handling
- Performance metrics collection

### Acceptance Criteria:
- Service starts and responds to health checks
- Can load and run inference on test models
- Achieves target performance benchmarks
- Includes comprehensive unit tests
            """,
            "labels": ["enhancement", "core-feature"]
        },
        {
            "title": "🔒 Security Audit and Hardening",
            "body": """## Security Audit and Hardening

### Security Review Areas:
- [ ] Code security analysis (Bandit, CodeQL)
- [ ] Dependency vulnerability scanning
- [ ] Container security hardening
- [ ] Kubernetes security policies
- [ ] API authentication and authorization
- [ ] Secrets management
- [ ] Network security policies

### Deliverables:
- Security audit report
- Remediation plan for findings
- Security documentation
- Penetration testing results

### Acceptance Criteria:
- Zero critical security vulnerabilities
- All security workflows pass
- Security documentation complete
- Incident response plan documented
            """,
            "labels": ["security", "urgent"]
        }
    ]
    
    for issue in issues:
        command = f'''gh issue create \\
            --title "{issue['title']}" \\
            --body "{issue['body']}" \\
            --label "{','.join(issue['labels'])}"'''
        
        result = run_command(command, check=False)
        if result:
            print(f"  ✅ Created issue: {issue['title']}")
        else:
            print(f"  ⚠️ Failed to create issue: {issue['title']}")

if __name__ == "__main__":
    print("🚀 VORTA GitHub Repository Setup")
    print("=" * 50)
    
    success = setup_github_repository()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 Repository setup completed!")
        print("\nNext steps:")
        print("1. 🔐 Add repository secrets manually in GitHub Settings")
        print("2. 👥 Invite team members and set up code owners")
        print("3. 🚀 Create first pull request to test workflows")
        print("4. 📋 Review and assign created issues")
        print("\n💡 Tip: Run 'gh repo view --web' to open repository in browser")
    else:
        print("\n❌ Repository setup failed!")
        print("Please check the errors above and try again.")
        sys.exit(1)
