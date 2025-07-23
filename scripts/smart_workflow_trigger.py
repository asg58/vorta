#!/usr/bin/env python3
"""
🚀 VORTA Smart Workflow Trigger
Auto-trigger van GitHub workflows met chat session integratie

Functionaliteit:
- Detecteert code wijzigingen
- Triggered intelligente workflow selectie
- Integreert chat session persistence
- Genereert workflow execution matrix

Author: VORTA Development Team
Version: 1.0.0 - Smart Triggering
"""

import os
import sys
from pathlib import Path


def main():
    """Main workflow trigger function"""
    print("🚀 VORTA Smart Workflow Trigger")
    print("=" * 50)
    
    # Check if we're in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("🤖 GitHub Actions environment detected")
        
        # Import and run intelligence engine
        sys.path.append(str(Path(__file__).parent))
        from github_workflow_intelligence import GitHubActionsIntelligence
        
        intelligence = GitHubActionsIntelligence()
        intelligence.execute_intelligent_dispatch()
        
    else:
        print("💻 Local development environment")
        print("   Run from GitHub Actions for full functionality")
        
        # Show what would happen
        sys.path.append(str(Path(__file__).parent))
        from github_workflow_intelligence import GitHubActionsIntelligence
        
        intelligence = GitHubActionsIntelligence()
        analysis = intelligence.analyze_changes()
        matrix = intelligence.create_workflow_matrix(analysis)
        
        print("\n🔍 Would trigger:")
        for workflow in matrix['workflows']:
            priority = matrix['priorities'].get(workflow, 'normal')
            print(f"   • {workflow} (priority: {priority})")
        
        print(f"\n📊 Session integration: {'✅' if matrix['session_integration'] else '❌'}")

if __name__ == "__main__":
    main()
