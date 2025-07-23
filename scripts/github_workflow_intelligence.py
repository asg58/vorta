#!/usr/bin/env python3
"""
🤖 VORTA GitHub Actions Intelligence Engine
Automatische workflow dispatching met chat session integratie

Features:
- Intelligente workflow selectie
- Chat session persistentie in CI/CD
- Automatische deployment van session systeem
- Performance monitoring in GitHub Actions
- Smart artifact management

Author: VORTA Development Team  
Version: 2.0.0 - GitHub Actions Integration
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class GitHubActionsIntelligence:
    """
    🤖 Intelligente GitHub Actions workflow manager
    
    Integreert chat session persistence in alle CI/CD workflows
    """
    
    # Workflow constants
    CHAT_SESSION_WORKFLOW = 'chat-session-automation.yml'
    STAGING_WORKFLOW = 'cd-staging.yml'
    PERFORMANCE_WORKFLOW = 'performance-test.yml'
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.github_context = self._load_github_context()
        self.workflow_decisions = []
        
        # Session integration config
        self.session_features = {
            'chat_persistence': True,
            'auto_restore': True,
            'vs_code_integration': True,
            'performance_monitoring': True,
            'cross_platform_support': True
        }
        
    def _load_github_context(self) -> Dict[str, Any]:
        """Load GitHub Actions context"""
        context = {
            'event_name': os.getenv('GITHUB_EVENT_NAME', 'unknown'),
            'ref': os.getenv('GITHUB_REF', 'refs/heads/main'),
            'sha': os.getenv('GITHUB_SHA', 'unknown'),
            'actor': os.getenv('GITHUB_ACTOR', 'unknown'),
            'repository': os.getenv('GITHUB_REPOSITORY', 'asg58/vorta'),
            'workflow': os.getenv('GITHUB_WORKFLOW', 'unknown'),
            'run_id': os.getenv('GITHUB_RUN_ID', 'unknown'),
            'run_number': os.getenv('GITHUB_RUN_NUMBER', 'unknown')
        }
        
        return context
    
    def _get_changed_files(self) -> List[str]:
        """Get list of changed files"""
        try:
            result = subprocess.run([
                'git', 'diff', '--name-only', 'HEAD~1', 'HEAD'
            ], capture_output=True, text=True)
            
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except Exception:
            return []
    
    def _categorize_changes(self, changed_files: List[str]) -> Dict[str, bool]:
        """Categorize changed files"""
        categories = {
            'frontend': False,
            'backend': False,
            'ai_components': False,
            'chat_session': False,
            'infrastructure': False,
            'docs': False
        }
        
        for file in changed_files:
            if file.startswith('frontend/'):
                categories['frontend'] = True
            elif file.startswith('services/'):
                categories['backend'] = True
            elif 'components/ai/' in file:
                categories['ai_components'] = True
            elif 'chat_session_persistence' in file or 'vorta_auto_startup' in file:
                categories['chat_session'] = True
            elif file.startswith('infrastructure/') or file.startswith('.github/'):
                categories['infrastructure'] = True
            elif file.startswith('docs/') or file.endswith('.md'):
                categories['docs'] = True
        
        return categories
    
    def _determine_triggers(self, categories: Dict[str, bool]) -> Dict[str, bool]:
        """Determine workflow triggers based on categories"""
        triggers = {
            'full_ci': False,
            'session_test': False,
            'performance_test': False,
            'security_scan': False,
            'deployment': False
        }
        
        if categories['chat_session']:
            triggers['session_test'] = True
            triggers['performance_test'] = True
            
        if categories['ai_components'] or categories['frontend']:
            triggers['full_ci'] = True
            
        if categories['backend'] or categories['infrastructure']:
            triggers['security_scan'] = True
            triggers['deployment'] = True
            
        return triggers

    def analyze_changes(self) -> Dict[str, Any]:
        """Analyseer wijzigingen om workflow selectie te bepalen"""
        changed_files = self._get_changed_files()
        categories = self._categorize_changes(changed_files)
        triggers = self._determine_triggers(categories)
        
        return {
            'changed_files': changed_files,
            'categories': categories,
            'triggers': triggers
        }
    
    def create_workflow_matrix(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent workflow execution matrix"""
        
        matrix = {
            'workflows': [],
            'priorities': {},
            'dependencies': {},
            'session_integration': True
        }
        
        # Always include chat session validation if session files changed
        if analysis['triggers'].get('session_test', False):
            matrix['workflows'].extend([
                self.CHAT_SESSION_WORKFLOW,
                self.PERFORMANCE_WORKFLOW
            ])
            matrix['priorities'][self.CHAT_SESSION_WORKFLOW] = 'high'
            
        # Add CI workflows based on changes
        if analysis['triggers'].get('full_ci', False):
            matrix['workflows'].extend([
                'ci.yml',
                'advanced-ci.yml',
                'enterprise-quality-gates.yml'
            ])
            
        # Add security and deployment
        if analysis['triggers'].get('security_scan', False):
            matrix['workflows'].append('security-scan.yml')
            
        if analysis['triggers'].get('deployment', False) and self.github_context['ref'] == 'refs/heads/main':
            matrix['workflows'].extend([
                self.STAGING_WORKFLOW,
                'smart-deployment.yml'
            ])
            
        # Set dependencies
        matrix['dependencies'] = {
            self.STAGING_WORKFLOW: ['ci.yml', 'security-scan.yml'],
            'smart-deployment.yml': [self.CHAT_SESSION_WORKFLOW, self.PERFORMANCE_WORKFLOW],
            'cd-production.yml': [self.STAGING_WORKFLOW]
        }
        
        return matrix
    
    def inject_session_context(self, workflow_path: str) -> str:
        """Inject chat session context into workflow"""
        try:
            with open(workflow_path, 'r') as f:
                workflow_content = f.read()
            
            # Add session environment variables
            session_env = """
      VORTA_SESSION_PERSISTENCE: 'true'
      VORTA_AUTO_RESTORE: 'true'
      VORTA_GITHUB_INTEGRATION: 'true'
"""
            
            # Insert after existing env section or create new one
            if 'env:' in workflow_content:
                workflow_content = workflow_content.replace(
                    'env:', f'env:{session_env}'
                )
            else:
                # Add after on: section
                workflow_content = workflow_content.replace(
                    'on:', f'env:{session_env}\n\non:'
                )
            
            return workflow_content
            
        except Exception as e:
            print(f"❌ Session context injection failed for {workflow_path}: {e}")
            return ""
    
    def create_dynamic_session_job(self) -> str:
        """Create dynamic session testing job"""
        return """
  # Dynamic Chat Session Integration
  chat-session-integration:
    name: 🔄 Dynamic Session Integration
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.modified, 'chat_session') || contains(github.event.head_commit.modified, 'vorta_auto_startup')
    
    steps:
      - name: 📥 Checkout
        uses: actions/checkout@v4
        
      - name: 🐍 Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
          
      - name: 🧠 Test Session Persistence
        run: |
          echo "🔄 Testing dynamic session integration..."
          python -c "
          import asyncio
          import sys
          sys.path.append('.')
          
          async def dynamic_session_test():
              try:
                  from frontend.components.ai.chat_session_persistence import ChatSessionPersistenceManager
                  
                  manager = ChatSessionPersistenceManager()
                  session_id = await manager.create_new_session(
                      title='Dynamic GitHub Test',
                      auto_restore=True
                  )
                  
                  print(f'✅ Dynamic session created: {session_id}')
                  
                  # Test integration with GitHub context
                  await manager.add_message(
                      session_id, 
                      'system', 
                      f'GitHub Actions run: ${{ github.run_id }}'
                  )
                  
                  metrics = manager.get_performance_metrics()
                  print(f'📊 Session metrics: {metrics}')
                  
                  return True
                  
              except Exception as e:
                  print(f'❌ Dynamic test failed: {e}')
                  return False
          
          success = asyncio.run(dynamic_session_test())
          exit(0 if success else 1)
          "
"""
    
    def execute_intelligent_dispatch(self):
        """Execute intelligent workflow dispatching"""
        print("🤖 VORTA GitHub Actions Intelligence Engine")
        print("=" * 60)
        
        # Analyze changes
        print("🔍 Analyzing repository changes...")
        analysis = self.analyze_changes()
        
        print(f"📁 Changed files: {len(analysis['changed_files'])}")
        for category, changed in analysis['categories'].items():
            if changed:
                print(f"   ✅ {category.replace('_', ' ').title()}")
        
        # Create workflow matrix
        print("\n🎯 Creating workflow execution matrix...")
        matrix = self.create_workflow_matrix(analysis)
        
        print(f"🚀 Workflows to execute: {len(matrix['workflows'])}")
        for workflow in matrix['workflows']:
            priority = matrix['priorities'].get(workflow, 'normal')
            print(f"   • {workflow} (priority: {priority})")
        
        # Generate execution plan
        print("\n📋 Generating execution plan...")
        execution_plan = {
            'github_context': self.github_context,
            'analysis': analysis,
            'workflow_matrix': matrix,
            'session_features': self.session_features,
            'execution_time': datetime.now().isoformat(),
            'intelligence_version': '2.0.0'
        }
        
        # Save execution plan
        plan_file = Path('.github/workflow-execution-plan.json')
        with open(plan_file, 'w') as f:
            json.dump(execution_plan, f, indent=2)
        
        print(f"💾 Execution plan saved: {plan_file}")
        
        # Create dynamic workflow enhancements
        if analysis['categories']['chat_session']:
            self._create_dynamic_session_workflow()
        
        # Output for GitHub Actions
        self._set_github_outputs(matrix, analysis)
        
        print("\n✅ Intelligent dispatch completed!")
        
    def _create_dynamic_session_workflow(self):
        """Create dynamic session workflow file"""
        dynamic_workflow = f"""name: 🔄 Dynamic Session Workflow

on:
  workflow_call:
    inputs:
      session_test_mode:
        required: false
        type: string
        default: 'full'

jobs:
{self.create_dynamic_session_job()}
  
  session-performance-benchmark:
    name: ⚡ Session Performance Benchmark  
    runs-on: ubuntu-latest
    needs: chat-session-integration
    
    steps:
      - name: 📥 Checkout
        uses: actions/checkout@v4
        
      - name: 🐍 Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
          
      - name: ⚡ Performance Benchmark
        run: |
          echo "⚡ Running session performance benchmark..."
          python -c "
          import time
          import asyncio
          import sys
          sys.path.append('.')
          
          async def performance_benchmark():
              from frontend.components.ai.chat_session_persistence import ChatSessionPersistenceManager
              
              manager = ChatSessionPersistenceManager()
              
              # Benchmark session creation
              start = time.time()
              session_ids = []
              for i in range(10):
                  session_id = await manager.create_new_session(title=f'Benchmark {{i}}')
                  session_ids.append(session_id)
              creation_time = (time.time() - start) * 1000
              
              print(f'📊 10 sessions created in {{creation_time:.1f}}ms')
              print(f'📊 Average per session: {{creation_time/10:.1f}}ms')
              
              # Benchmark message addition
              start = time.time()
              for session_id in session_ids:
                  await manager.add_message(session_id, 'user', 'Benchmark message')
              message_time = (time.time() - start) * 1000
              
              print(f'📊 10 messages added in {{message_time:.1f}}ms')
              
              # Benchmark backup
              start = time.time()
              await manager.backup_all_sessions()
              backup_time = (time.time() - start) * 1000
              
              print(f'📊 Session backup completed in {{backup_time:.1f}}ms')
              
              # Performance thresholds
              if creation_time/10 < 100 and message_time/10 < 50 and backup_time < 500:
                  print('✅ All performance benchmarks passed!')
                  return True
              else:
                  print('⚠️ Some performance benchmarks did not meet targets')
                  return False
          
          success = asyncio.run(performance_benchmark())
          exit(0 if success else 1)
          "
"""
        
        dynamic_file = Path('.github/workflows/dynamic-session-workflow.yml')
        with open(dynamic_file, 'w') as f:
            f.write(dynamic_workflow)
        
        print(f"🔄 Dynamic session workflow created: {dynamic_file}")
    
    def _set_github_outputs(self, matrix: Dict[str, Any], analysis: Dict[str, Any]):
        """Set GitHub Actions outputs"""
        if os.getenv('GITHUB_ACTIONS'):
            # Set outputs for GitHub Actions
            outputs = {
                'workflows': ','.join(matrix['workflows']),
                'session_test_required': str(analysis['triggers'].get('session_test', False)).lower(),
                'performance_test_required': str(analysis['triggers'].get('performance_test', False)).lower(),
                'deployment_required': str(analysis['triggers'].get('deployment', False)).lower(),
                'session_integration': str(matrix['session_integration']).lower()
            }
            
            for key, value in outputs.items():
                print(f"::set-output name={key}::{value}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--github-mode':
        # GitHub Actions mode
        intelligence = GitHubActionsIntelligence()
        intelligence.execute_intelligent_dispatch()
    else:
        # Local development mode
        print("🤖 VORTA GitHub Actions Intelligence Engine")
        print("📝 Run with --github-mode for GitHub Actions integration")
        
        intelligence = GitHubActionsIntelligence()
        analysis = intelligence.analyze_changes()
        matrix = intelligence.create_workflow_matrix(analysis)
        
        print("\n🔍 Analysis Results:")
        print(f"   Changed files: {len(analysis['changed_files'])}")
        print(f"   Workflows suggested: {len(matrix['workflows'])}")
        print(f"   Session integration: {matrix['session_integration']}")

if __name__ == "__main__":
    main()
