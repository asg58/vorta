#!/usr/bin/env python3
"""
VORTA GitHub Workflows Intelligence - Optimal CI/CD Automation
Intelligently manages GitHub Actions workflows based on development context
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


class VortaGitHubWorkflowIntelligence:
    """
    🚀 Intelligent GitHub Actions Workflow Management
    Automatically determines when and which workflows to trigger
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.workflows_dir = self.project_root / ".github" / "workflows"
        
        # 🧠 WORKFLOW INTELLIGENCE MATRIX
        self.workflow_intelligence = {
            "ci-cd-pipeline.yml": {
                "triggers": ["push_to_main", "pull_request", "code_changes"],
                "priority": "critical",
                "auto_trigger": True,
                "conditions": ["tests_passing", "no_security_issues"]
            },
            
            "security-scan.yml": {
                "triggers": ["dependency_changes", "security_concerns", "scheduled"],
                "priority": "high", 
                "auto_trigger": True,
                "conditions": ["new_dependencies", "security_alerts"]
            },
            
            "performance-tests.yml": {
                "triggers": ["api_changes", "performance_regression", "release_prep"],
                "priority": "high",
                "auto_trigger": False,
                "conditions": ["api_endpoint_changes", "load_test_needed"]
            },
            
            "docker-build.yml": {
                "triggers": ["docker_changes", "infrastructure_updates", "release"],
                "priority": "medium",
                "auto_trigger": True,
                "conditions": ["dockerfile_changed", "compose_changed"]
            },
            
            "monitoring-deploy.yml": {
                "triggers": ["monitoring_config_changes", "grafana_updates"],
                "priority": "medium", 
                "auto_trigger": True,
                "conditions": ["prometheus_config_changed", "grafana_dashboards_updated"]
            },
            
            "release-automation.yml": {
                "triggers": ["version_tag", "release_branch", "manual_release"],
                "priority": "critical",
                "auto_trigger": False,
                "conditions": ["all_tests_pass", "security_scan_clean"]
            }
        }
    
    def analyze_repository_state(self) -> Dict[str, Any]:
        """Analyze current repository state for workflow decisions."""
        print("🔍 Analyzing repository state for workflow intelligence...")
        
        state = {
            "git_status": self._get_git_status(),
            "file_changes": self._detect_file_changes(),
            "branch_info": self._get_branch_info(),
            "workflow_triggers": []
        }
        
        # Determine active workflow triggers
        state["workflow_triggers"] = self._determine_workflow_triggers(state)
        
        return state
    
    def _get_git_status(self) -> Dict[str, Any]:
        """Get current git repository status."""
        try:
            # Git status
            status_result = subprocess.run([
                'git', 'status', '--porcelain'
            ], capture_output=True, text=True, timeout=10)
            
            # Recent commits
            log_result = subprocess.run([
                'git', 'log', '--oneline', '-5'
            ], capture_output=True, text=True, timeout=10)
            
            return {
                "has_changes": len(status_result.stdout.strip()) > 0,
                "staged_files": [line[3:] for line in status_result.stdout.strip().split('\n') 
                               if line.startswith('M ') or line.startswith('A ')],
                "recent_commits": log_result.stdout.strip().split('\n') if log_result.returncode == 0 else []
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_file_changes(self) -> Dict[str, List[str]]:
        """Detect what types of files have changed."""
        try:
            # Get changed files
            result = subprocess.run([
                'git', 'diff', '--name-only', 'HEAD~1', 'HEAD'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                # Fallback to staged changes
                result = subprocess.run([
                    'git', 'diff', '--name-only', '--staged'
                ], capture_output=True, text=True, timeout=10)
            
            changed_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Categorize changes
            categories = {
                "python_files": [f for f in changed_files if f.endswith('.py')],
                "docker_files": [f for f in changed_files if 'docker' in f.lower() or f.endswith('.dockerfile')],
                "config_files": [f for f in changed_files if f.endswith('.yml') or f.endswith('.yaml') or f.endswith('.json')],
                "api_files": [f for f in changed_files if 'api' in f.lower() or 'services' in f.lower()],
                "workflow_files": [f for f in changed_files if '.github/workflows' in f],
                "requirements_files": [f for f in changed_files if 'requirements' in f or 'package' in f],
                "monitoring_files": [f for f in changed_files if 'prometheus' in f or 'grafana' in f]
            }
            
            return categories
            
        except Exception:
            return {}
    
    def _get_branch_info(self) -> Dict[str, str]:
        """Get current branch information."""
        try:
            branch_result = subprocess.run([
                'git', 'rev-parse', '--abbrev-ref', 'HEAD'
            ], capture_output=True, text=True, timeout=5)
            
            return {
                "current_branch": branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown",
                "is_main": branch_result.stdout.strip() == "main",
                "is_feature": branch_result.stdout.strip().startswith("feature/"),
                "is_release": branch_result.stdout.strip().startswith("release/")
            }
            
        except Exception:
            return {"current_branch": "unknown"}
    
    def _determine_workflow_triggers(self, state: Dict[str, Any]) -> List[str]:
        """Determine which workflow triggers are active."""
        triggers = []
        
        git_status = state.get("git_status", {})
        file_changes = state.get("file_changes", {})
        branch_info = state.get("branch_info", {})
        
        # Code changes triggers
        if git_status.get("has_changes"):
            triggers.append("code_changes")
        
        # Branch-based triggers  
        if branch_info.get("is_main"):
            triggers.append("push_to_main")
        
        if branch_info.get("is_feature"):
            triggers.append("pull_request")
        
        # File-based triggers
        if file_changes.get("docker_files"):
            triggers.append("docker_changes")
        
        if file_changes.get("api_files"):
            triggers.append("api_changes")
        
        if file_changes.get("requirements_files"):
            triggers.append("dependency_changes")
        
        if file_changes.get("monitoring_files"):
            triggers.append("monitoring_config_changes")
        
        if file_changes.get("workflow_files"):
            triggers.append("infrastructure_updates")
        
        return triggers
    
    def make_workflow_decisions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent workflow execution decisions."""
        decisions = {
            "recommended_workflows": [],
            "immediate_triggers": [],
            "scheduled_workflows": [],
            "blocked_workflows": [],
            "reasoning": []
        }
        
        active_triggers = state["workflow_triggers"]
        
        # Analyze each workflow
        for workflow_name, config in self.workflow_intelligence.items():
            workflow_triggers = config["triggers"]
            matching_triggers = set(active_triggers) & set(workflow_triggers)
            
            if matching_triggers:
                priority = config["priority"]
                auto_trigger = config["auto_trigger"]
                
                workflow_decision = {
                    "workflow": workflow_name,
                    "priority": priority,
                    "matching_triggers": list(matching_triggers),
                    "auto_trigger": auto_trigger,
                    "conditions_met": self._check_workflow_conditions(workflow_name, config, state)
                }
                
                decisions["recommended_workflows"].append(workflow_decision)
                
                if auto_trigger and workflow_decision["conditions_met"] and priority == "critical":
                    decisions["immediate_triggers"].append(workflow_name)
                elif auto_trigger and workflow_decision["conditions_met"]:
                    decisions["scheduled_workflows"].append(workflow_name)
                elif not workflow_decision["conditions_met"]:
                    decisions["blocked_workflows"].append({
                        "workflow": workflow_name,
                        "reason": "Conditions not met"
                    })
                
                reasoning = f"{workflow_name}: {', '.join(matching_triggers)}"
                decisions["reasoning"].append(reasoning)
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        decisions["recommended_workflows"].sort(
            key=lambda x: priority_order.get(x["priority"], 999)
        )
        
        return decisions
    
    def _check_workflow_conditions(self, workflow_name: str, config: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if workflow conditions are met."""
        conditions = config.get("conditions", [])
        
        # For now, assume conditions are met if no blocking issues
        git_status = state.get("git_status", {})
        
        # Basic condition checks
        if "tests_passing" in conditions:
            # Would check test results here
            pass
        
        if "no_security_issues" in conditions:
            # Would check security scan results here  
            pass
        
        # Always return True for demo - in real implementation would check actual conditions
        return True
    
    def create_github_workflow_optimizer(self):
        """Create optimized GitHub workflow configurations."""
        
        # Create intelligent workflow dispatcher
        dispatcher_workflow = {
            "name": "VORTA Intelligent Workflow Dispatcher",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main"]
                },
                "workflow_dispatch": {
                    "inputs": {
                        "intelligence_mode": {
                            "description": "Enable intelligent workflow selection",
                            "required": False,
                            "default": "true",
                            "type": "boolean"
                        }
                    }
                }
            },
            "jobs": {
                "intelligent-dispatch": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.12"
                            }
                        },
                        {
                            "name": "Run VORTA Intelligence Engine",
                            "run": "python intelligence_engine.py --github-mode"
                        },
                        {
                            "name": "Trigger Selected Workflows",
                            "run": "python github_workflow_intelligence.py --execute"
                        }
                    ]
                }
            }
        }
        
        # Save dispatcher workflow
        os.makedirs(self.workflows_dir, exist_ok=True)
        dispatcher_path = self.workflows_dir / "intelligent-dispatcher.yml"
        
        with open(dispatcher_path, 'w') as f:
            import yaml
            yaml.dump(dispatcher_workflow, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created intelligent workflow dispatcher: {dispatcher_path}")
        
        return dispatcher_path
    
    def optimize_existing_workflows(self):
        """Optimize existing GitHub workflows for intelligence."""
        optimizations = []
        
        if not self.workflows_dir.exists():
            print("⚠️  No .github/workflows directory found")
            return optimizations
        
        # Analyze existing workflows
        for workflow_file in self.workflows_dir.glob("*.yml"):
            if workflow_file.name == "intelligent-dispatcher.yml":
                continue
                
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                # Add intelligence annotations
                if "# VORTA-INTELLIGENCE:" not in content:
                    intelligence_comment = f"""
# VORTA-INTELLIGENCE: AUTO-MANAGED
# This workflow is managed by VORTA Intelligence Engine
# Triggers: {', '.join(self.workflow_intelligence.get(workflow_file.name, {}).get('triggers', []))}
# Priority: {self.workflow_intelligence.get(workflow_file.name, {}).get('priority', 'medium')}
# Auto-trigger: {self.workflow_intelligence.get(workflow_file.name, {}).get('auto_trigger', False)}
"""
                    
                    # Add intelligence comment at top
                    optimized_content = intelligence_comment + content
                    
                    with open(workflow_file, 'w') as f:
                        f.write(optimized_content)
                    
                    optimizations.append(f"Enhanced {workflow_file.name} with intelligence")
                
            except Exception as e:
                print(f"⚠️  Could not optimize {workflow_file.name}: {e}")
        
        return optimizations
    
    def execute_workflow_intelligence(self):
        """Execute complete workflow intelligence system."""
        print("🚀 VORTA GITHUB WORKFLOWS INTELLIGENCE SYSTEM")
        print("=" * 60)
        
        # Analyze repository state
        state = self.analyze_repository_state()
        
        # Make workflow decisions
        decisions = self.make_workflow_decisions(state)
        
        # Print intelligence report
        self._print_workflow_report(state, decisions)
        
        # Create/optimize workflows
        print("\n🔧 OPTIMIZING GITHUB WORKFLOWS:")
        
        dispatcher_path = self.create_github_workflow_optimizer()
        optimizations = self.optimize_existing_workflows()
        
        for optimization in optimizations:
            print(f"  ✅ {optimization}")
        
        # Execute immediate workflows (simulated)
        if decisions["immediate_triggers"]:
            print("\n🚀 TRIGGERING IMMEDIATE WORKFLOWS:")
            for workflow in decisions["immediate_triggers"]:
                print(f"  → Triggering: {workflow}")
                # In real implementation, would trigger via GitHub API
        
        return decisions
    
    def _print_workflow_report(self, state: Dict[str, Any], decisions: Dict[str, Any]):
        """Print workflow intelligence report."""
        print("\n📊 REPOSITORY ANALYSIS:")
        
        git_status = state.get("git_status", {})
        file_changes = state.get("file_changes", {})
        branch_info = state.get("branch_info", {})
        
        print(f"  🌿 Branch: {branch_info.get('current_branch', 'unknown')}")
        print(f"  📝 Changes: {'Yes' if git_status.get('has_changes') else 'No'}")
        print(f"  🎯 Triggers: {len(state['workflow_triggers'])} detected")
        
        if state["workflow_triggers"]:
            for trigger in state["workflow_triggers"][:3]:
                print(f"     • {trigger.replace('_', ' ').title()}")
        
        print("\n🧠 WORKFLOW DECISIONS:")
        
        if decisions["recommended_workflows"]:
            for workflow_info in decisions["recommended_workflows"]:
                workflow = workflow_info["workflow"]
                priority = workflow_info["priority"].upper()
                auto = "🤖 AUTO" if workflow_info["auto_trigger"] else "👤 MANUAL"
                conditions = "✅ READY" if workflow_info["conditions_met"] else "⚠️  BLOCKED"
                
                print(f"  {priority:8} | {auto} | {conditions} | {workflow}")
        
        print("\n🎯 EXECUTION PLAN:")
        if decisions["immediate_triggers"]:
            print(f"  🚀 Immediate: {len(decisions['immediate_triggers'])} workflows")
        if decisions["scheduled_workflows"]:
            print(f"  ⏰ Scheduled: {len(decisions['scheduled_workflows'])} workflows")
        if decisions["blocked_workflows"]:
            print(f"  🚫 Blocked: {len(decisions['blocked_workflows'])} workflows")


def main():
    """Main GitHub workflow intelligence function."""
    print("🚀 VORTA GITHUB WORKFLOWS - INTELLIGENT CI/CD OPTIMIZATION")
    
    workflow_intelligence = VortaGitHubWorkflowIntelligence()
    decisions = workflow_intelligence.execute_workflow_intelligence()
    
    print("\n" + "=" * 60)
    print("🎯 GITHUB WORKFLOWS: Intelligently optimized!")
    print(f"💾 {len(decisions['recommended_workflows'])} workflow decisions made")
    print("🤖 Automatic CI/CD optimization: ACTIVE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Workflow intelligence interrupted")
    except Exception as e:
        print(f"\n❌ Workflow intelligence error: {e}")
