# VORTA GitHub Actions Setup Guide

## 🚀 Overview

This guide explains how to set up the required GitHub secrets and variables for the VORTA CI/CD pipeline.

## 📋 Required Secrets

### Production Secrets

Add these secrets in GitHub: Settings → Secrets and variables → Actions → Repository secrets

| Secret Name               | Description                                     | Example                                     |
| ------------------------- | ----------------------------------------------- | ------------------------------------------- |
| `PRODUCTION_DATABASE_URL` | Production PostgreSQL connection string         | `postgresql://user:pass@prod-db:5432/vorta` |
| `PRODUCTION_API_KEY`      | Production API authentication key               | `prod-api-key-12345`                        |
| `PRODUCTION_REDIS_URL`    | Production Redis connection string              | `redis://prod-redis:6379/0`                 |
| `PRODUCTION_JWT_SECRET`   | Production JWT signing secret                   | `super-secret-jwt-key-production`           |
| `KUBE_CONFIG_PRODUCTION`  | Base64 encoded Kubernetes config for production | `base64-encoded-kubeconfig`                 |

### Staging Secrets

| Secret Name            | Description                                  | Example                                        |
| ---------------------- | -------------------------------------------- | ---------------------------------------------- |
| `STAGING_DATABASE_URL` | Staging PostgreSQL connection string         | `postgresql://user:pass@staging-db:5432/vorta` |
| `STAGING_API_KEY`      | Staging API authentication key               | `staging-api-key-12345`                        |
| `STAGING_REDIS_URL`    | Staging Redis connection string              | `redis://staging-redis:6379/1`                 |
| `STAGING_JWT_SECRET`   | Staging JWT signing secret                   | `staging-jwt-secret`                           |
| `KUBE_CONFIG_STAGING`  | Base64 encoded Kubernetes config for staging | `base64-encoded-kubeconfig`                    |

### Optional Secrets

| Secret Name     | Description                | Default              |
| --------------- | -------------------------- | -------------------- |
| `BACKUP_BUCKET` | S3 bucket name for backups | `vorta-backups-demo` |

## 🌐 Required Variables

Add these variables in GitHub: Settings → Secrets and variables → Actions → Repository variables

| Variable Name         | Description                 | Example                                 |
| --------------------- | --------------------------- | --------------------------------------- |
| `PRODUCTION_ENDPOINT` | Production API endpoint URL | `https://api.vorta.example.com`         |
| `STAGING_ENDPOINT`    | Staging API endpoint URL    | `https://staging-api.vorta.example.com` |

## 🔧 Setup Steps

### 1. Add Repository Secrets

```bash
# Using GitHub CLI
gh secret set PRODUCTION_DATABASE_URL --body "postgresql://user:pass@prod-db:5432/vorta"
gh secret set PRODUCTION_API_KEY --body "your-production-api-key"
gh secret set STAGING_DATABASE_URL --body "postgresql://user:pass@staging-db:5432/vorta"
gh secret set STAGING_API_KEY --body "your-staging-api-key"
```

### 2. Add Repository Variables

```bash
# Using GitHub CLI
gh variable set PRODUCTION_ENDPOINT --body "https://api.vorta.example.com"
gh variable set STAGING_ENDPOINT --body "https://staging-api.vorta.example.com"
```

### 3. Kubernetes Configuration

To get base64 encoded kubeconfig:

```bash
# Encode your kubeconfig
cat ~/.kube/config | base64 -w 0
```

Then add the output as `KUBE_CONFIG_PRODUCTION` and `KUBE_CONFIG_STAGING` secrets.

## 🛡️ Default Values

If secrets/variables are not set, the workflows will use these default values:

- **Database URLs**: `postgresql://localhost:5432/vorta_prod` (or `vorta_staging`)
- **API Keys**: `demo-api-key` (or `demo-staging-key`)
- **Redis URLs**: `redis://localhost:6379/0` (or `/1`)
- **Endpoints**: `https://api.vorta.example.com` (or `https://staging-api.vorta.example.com`)
- **JWT Secrets**: Demo secrets (change in production!)

## ⚠️ VS Code Warnings

You may see "Context access might be invalid" warnings in VS Code for secrets/variables that don't exist yet. These warnings are normal and don't prevent the workflows from running.

## 🧪 Testing Workflows

Once secrets are configured, you can test the workflows:

1. **Setup & Verification**: Run the "VORTA Setup & Verification" workflow
2. **CI Pipeline**: Push code to trigger the continuous integration workflow
3. **Security Scan**: Run the security scanning workflow
4. **Performance Test**: Run performance tests manually

## 🔄 Workflow Status

The VORTA project includes these GitHub Actions workflows:

- ✅ **CI (Continuous Integration)** - Automated testing and validation
- ✅ **Security Scan** - Multi-layer security analysis
- ✅ **CD Staging** - Automated staging deployment
- ✅ **CD Production** - Production deployment with approval gates
- ✅ **Dependency Updates** - Automated dependency management
- ✅ **Performance Testing** - Load and performance validation
- ✅ **Setup & Verification** - Project structure validation

## 📚 Additional Resources

- [GitHub Actions Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub Actions Variables Documentation](https://docs.github.com/en/actions/learn-github-actions/variables)
- [VORTA Project Documentation](./VORTA_Implementation_Guide.md)
