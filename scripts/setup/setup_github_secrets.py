#!/usr/bin/env python3
"""
VORTA GitHub Secrets Setup Script
Helps set up all required GitHub secrets and variables for VORTA workflows
"""

import subprocess
import sys


def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode

def check_gh_cli():
    """Check if GitHub CLI is installed and authenticated"""
    print("🔍 Checking GitHub CLI...")
    
    stdout, stderr, code = run_command("gh --version", check=False)
    if code != 0:
        print("❌ GitHub CLI is not installed!")
        print("   Install it from: https://cli.github.com/")
        return False
    
    print(f"✅ GitHub CLI found: {stdout.split()[2]}")
    
    # Check if authenticated
    stdout, stderr, code = run_command("gh auth status", check=False)
    if code != 0:
        print("❌ Not authenticated with GitHub!")
        print("   Run: gh auth login")
        return False
    
    print("✅ GitHub CLI authenticated")
    return True

def get_user_input(prompt, default=None, secret=False):
    """Get user input with optional default value"""
    if default:
        prompt += f" (default: {default})"
    prompt += ": "
    
    if secret:
        import getpass
        value = getpass.getpass(prompt)
    else:
        value = input(prompt)
    
    return value.strip() or default

def setup_secrets():
    """Set up GitHub repository secrets"""
    print("\n🔐 Setting up GitHub Secrets...")
    
    secrets = {
        "PRODUCTION_DATABASE_URL": {
            "description": "Production PostgreSQL connection string",
            "default": "postgresql://localhost:5432/vorta_prod",
            "secret": True
        },
        "STAGING_DATABASE_URL": {
            "description": "Staging PostgreSQL connection string", 
            "default": "postgresql://localhost:5432/vorta_staging",
            "secret": True
        },
        "PRODUCTION_API_KEY": {
            "description": "Production API key",
            "default": "prod-api-key-change-me",
            "secret": True
        },
        "STAGING_API_KEY": {
            "description": "Staging API key",
            "default": "staging-api-key-change-me", 
            "secret": True
        },
        "PRODUCTION_REDIS_URL": {
            "description": "Production Redis connection string",
            "default": "redis://localhost:6379/0",
            "secret": True
        },
        "STAGING_REDIS_URL": {
            "description": "Staging Redis connection string",
            "default": "redis://localhost:6379/1",
            "secret": True
        },
        "PRODUCTION_JWT_SECRET": {
            "description": "Production JWT signing secret",
            "default": "change-this-jwt-secret-in-production",
            "secret": True
        },
        "STAGING_JWT_SECRET": {
            "description": "Staging JWT signing secret", 
            "default": "staging-jwt-secret",
            "secret": True
        }
    }
    
    for secret_name, config in secrets.items():
        print(f"\n📝 {config['description']}")
        value = get_user_input(f"Enter {secret_name}", config['default'], config['secret'])
        
        if value:
            print(f"   Setting secret {secret_name}...")
            stdout, stderr, code = run_command(f'gh secret set {secret_name} --body "{value}" --repo asg58/ai-voice-agent-v3', check=False)
            
            if code == 0:
                print(f"   ✅ {secret_name} set successfully")
            else:
                print(f"   ❌ Failed to set {secret_name}: {stderr}")
        else:
            print(f"   ⏭️  Skipping {secret_name}")

def setup_variables():
    """Set up GitHub repository variables"""
    print("\n🌐 Setting up GitHub Variables...")
    
    variables = {
        "PRODUCTION_ENDPOINT": {
            "description": "Production API endpoint URL",
            "default": "https://api.vorta.example.com"
        },
        "STAGING_ENDPOINT": {
            "description": "Staging API endpoint URL",
            "default": "https://staging-api.vorta.example.com"
        }
    }
    
    for var_name, config in variables.items():
        print(f"\n📝 {config['description']}")
        value = get_user_input(f"Enter {var_name}", config['default'])
        
        if value:
            print(f"   Setting variable {var_name}...")
            stdout, stderr, code = run_command(f'gh variable set {var_name} --body "{value}" --repo asg58/ai-voice-agent-v3', check=False)
            
            if code == 0:
                print(f"   ✅ {var_name} set successfully")
            else:
                print(f"   ❌ Failed to set {var_name}: {stderr}")
        else:
            print(f"   ⏭️  Skipping {var_name}")

def setup_kubernetes():
    """Help user set up Kubernetes configuration secrets"""
    print("\n☸️  Setting up Kubernetes Configuration...")
    print("\nTo set up Kubernetes secrets, you need base64-encoded kubeconfig files.")
    print("You can generate them using: cat ~/.kube/config | base64 -w 0")
    
    setup_kube = input("\nDo you want to set up Kubernetes secrets now? (y/N): ").strip().lower()
    if setup_kube == 'y':
        print("\n📝 Production Kubernetes config (base64 encoded)")
        prod_kube = get_user_input("Enter KUBE_CONFIG_PRODUCTION", secret=True)
        
        if prod_kube:
            stdout, stderr, code = run_command(f'gh secret set KUBE_CONFIG_PRODUCTION --body "{prod_kube}" --repo asg58/ai-voice-agent-v3', check=False)
            if code == 0:
                print("   ✅ KUBE_CONFIG_PRODUCTION set successfully")
            else:
                print(f"   ❌ Failed to set KUBE_CONFIG_PRODUCTION: {stderr}")
        
        print("\n📝 Staging Kubernetes config (base64 encoded)")
        staging_kube = get_user_input("Enter KUBE_CONFIG_STAGING", secret=True)
        
        if staging_kube:
            stdout, stderr, code = run_command(f'gh secret set KUBE_CONFIG_STAGING --body "{staging_kube}" --repo asg58/ai-voice-agent-v3', check=False)
            if code == 0:
                print("   ✅ KUBE_CONFIG_STAGING set successfully")
            else:
                print(f"   ❌ Failed to set KUBE_CONFIG_STAGING: {stderr}")
    else:
        print("⏭️  Skipping Kubernetes configuration")

def verify_setup():
    """Verify that secrets and variables are set correctly"""
    print("\n🔍 Verifying setup...")
    
    # List secrets
    stdout, stderr, code = run_command("gh secret list --repo asg58/ai-voice-agent-v3", check=False)
    if code == 0:
        print("\n✅ Current repository secrets:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"   🔐 {line.split()[0]}")
    else:
        print("❌ Could not list secrets")
    
    # List variables
    stdout, stderr, code = run_command("gh variable list --repo asg58/ai-voice-agent-v3", check=False)
    if code == 0:
        print("\n✅ Current repository variables:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"   🌐 {line.split()[0]}")
    else:
        print("❌ Could not list variables")

def main():
    """Main setup function"""
    print("🚀 VORTA GitHub Actions Setup")
    print("=" * 40)
    
    if not check_gh_cli():
        return 1
    
    print("\nThis script will help you set up GitHub secrets and variables for VORTA workflows.")
    print("You can skip any value by pressing Enter (default values will be used).")
    
    proceed = input("\nDo you want to proceed? (Y/n): ").strip().lower()
    if proceed == 'n':
        print("Setup cancelled.")
        return 0
    
    # Set up secrets
    setup_secrets()
    
    # Set up variables
    setup_variables()
    
    # Set up Kubernetes (optional)
    setup_kubernetes()
    
    # Verify setup
    verify_setup()
    
    print("\n" + "=" * 40)
    print("✅ VORTA GitHub Actions setup completed!")
    print("🚀 Your workflows are now ready to run.")
    print("\nNext steps:")
    print("1. Push code to trigger CI pipeline")
    print("2. Run 'VORTA Setup & Verification' workflow")
    print("3. Test individual workflows as needed")
    print("\nFor more information, see GITHUB_ACTIONS_SETUP.md")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
