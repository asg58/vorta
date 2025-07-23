#!/usr/bin/env python3
"""
VORTA Intelligence Engine - Automatic Tool Decision System
PERMANENT MEMORY INJECTION: When to use which VORTA development tools
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class VortaIntelligenceEngine:
    """
    🧠 PERMANENT MEMORY: VORTA Development Tool Decision Matrix
    
    This system AUTOMATICALLY determines when to use each tool based on:
    - Current development context
    - System state
    - Time patterns
    - User behavior
    - Code changes
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.memory_file = self.project_root / "vorta_intelligence_memory.json"
        
        # 🧠 PERMANENT DECISION RULES - INJECTED INTO MEMORY
        self.tool_decision_matrix = {
            "smart_startup.py": {
                "triggers": [
                    "system_boot",           # After computer restart
                    "vscode_first_open",     # First VS Code session today
                    "services_down",         # When services not responding
                    "docker_stopped",        # When containers stopped
                    "development_start",     # Beginning of dev session
                    "after_lunch_break",     # After long inactivity
                ],
                "frequency": "session_start",
                "priority": "critical",
                "auto_execute": True
            },
            
            "test_metrics.py": {
                "triggers": [
                    "before_commit",         # Before git commit
                    "after_code_change",     # After API modifications
                    "pre_deployment",        # Before deployment
                    "api_endpoint_added",    # New endpoints created
                    "debugging_session",     # During debugging
                    "performance_concern",   # When performance issues
                ],
                "frequency": "on_demand",
                "priority": "high", 
                "auto_execute": False
            },
            
            "dev_dashboard.py": {
                "triggers": [
                    "active_development",    # During coding sessions
                    "monitoring_needed",     # When monitoring required
                    "performance_analysis",  # Performance debugging
                    "service_debugging",     # Service issues
                    "metrics_investigation", # Metrics analysis
                    "real_time_monitoring",  # Live development
                ],
                "frequency": "continuous",
                "priority": "medium",
                "auto_execute": False
            },
            
            "quick_check.py": {
                "triggers": [
                    "between_tasks",         # Between development tasks
                    "rapid_validation",      # Quick health verification
                    "ci_cd_check",          # CI/CD pipeline verification
                    "pre_meeting",          # Before demo/meetings
                    "deployment_verification", # After deployment
                    "troubleshooting_start", # Beginning troubleshooting
                ],
                "frequency": "frequent",
                "priority": "low",
                "auto_execute": True
            }
        }
        
        # 🎯 CONTEXT DETECTION PATTERNS
        self.context_patterns = {
            "development_active": [
                "recent_file_changes",
                "vscode_sessions_active", 
                "git_activity",
                "test_runs"
            ],
            "system_fresh_start": [
                "docker_containers_zero",
                "services_not_responding",
                "last_activity_old"
            ],
            "debugging_session": [
                "error_logs_recent",
                "service_failures",
                "performance_issues"
            ],
            "deployment_preparation": [
                "git_staged_changes",
                "test_failures",
                "version_tags"
            ]
        }
    
    def save_memory(self, memory_data: Dict[str, Any]):
        """💾 Save decision intelligence to persistent memory."""
        try:
            memory_data["last_updated"] = datetime.now().isoformat()
            memory_data["intelligence_version"] = "2.0"
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Memory save failed: {e}")
    
    def load_memory(self) -> Dict[str, Any]:
        """🧠 Load persistent intelligence memory."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Default memory state
        return {
            "last_smart_startup": None,
            "last_test_run": None,
            "last_dashboard_use": None,
            "last_quick_check": None,
            "development_sessions": [],
            "pattern_history": {},
            "auto_decisions": []
        }
    
    def detect_current_context(self) -> Dict[str, Any]:
        """🔍 Detect current development context automatically."""
        context = {
            "timestamp": datetime.now().isoformat(),
            "system_state": {},
            "development_activity": {},
            "service_health": {},
            "triggers_detected": []
        }
        
        # Check system state
        context["system_state"] = self._check_system_state()
        
        # Check development activity  
        context["development_activity"] = self._check_development_activity()
        
        # Check service health
        context["service_health"] = self._check_service_health()
        
        # Detect active triggers
        context["triggers_detected"] = self._detect_active_triggers(context)
        
        return context
    
    def _check_system_state(self) -> Dict[str, Any]:
        """Check system and Docker state."""
        try:
            # Check Docker containers
            docker_result = subprocess.run([
                'docker', 'ps', '--format', '{{.Names}}'
            ], capture_output=True, text=True, timeout=5)
            
            vorta_containers = []
            if docker_result.returncode == 0:
                for line in docker_result.stdout.strip().split('\n'):
                    if 'vorta' in line.lower():
                        vorta_containers.append(line.strip())
            
            return {
                "docker_containers": len(vorta_containers),
                "containers_list": vorta_containers,
                "docker_available": docker_result.returncode == 0
            }
            
        except Exception:
            return {"docker_containers": 0, "docker_available": False}
    
    def _check_development_activity(self) -> Dict[str, Any]:
        """Check recent development activity."""
        activity = {
            "recent_changes": False,
            "git_activity": False,
            "vscode_session_active": True  # Assume true since we're running
        }
        
        try:
            # Check git status
            git_result = subprocess.run([
                'git', 'status', '--porcelain'
            ], capture_output=True, text=True, timeout=5)
            
            if git_result.returncode == 0:
                activity["git_activity"] = len(git_result.stdout.strip()) > 0
                activity["recent_changes"] = activity["git_activity"]
            
        except Exception:
            pass
        
        return activity
    
    def _check_service_health(self) -> Dict[str, bool]:
        """Quick service health check."""
        services = {"api": 8000, "prometheus": 9090, "grafana": 3000}
        health = {}
        
        for service, port in services.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                health[service] = result == 0
                sock.close()
            except:
                health[service] = False
        
        return health
    
    def _detect_active_triggers(self, context: Dict[str, Any]) -> List[str]:
        """Detect which triggers are currently active."""
        triggers = []
        
        system_state = context["system_state"]
        dev_activity = context["development_activity"]
        service_health = context["service_health"]
        
        # System state triggers
        if system_state["docker_containers"] == 0:
            triggers.append("docker_stopped")
            triggers.append("system_fresh_start")
        
        if not all(service_health.values()):
            triggers.append("services_down")
        
        # Development activity triggers
        if dev_activity["recent_changes"]:
            triggers.append("after_code_change")
        
        if dev_activity["git_activity"]:
            triggers.append("before_commit")
        
        # Time-based triggers
        memory = self.load_memory()
        now = datetime.now()
        
        if not memory.get("last_smart_startup"):
            triggers.append("development_start")
        else:
            last_startup = datetime.fromisoformat(memory["last_smart_startup"])
            if (now - last_startup).total_seconds() > 28800:  # 8 hours
                triggers.append("development_start")
        
        return triggers
    
    def make_intelligent_decisions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """🧠 Make intelligent decisions about which tools to use."""
        decisions = {
            "recommended_tools": [],
            "immediate_actions": [],
            "scheduled_actions": [],
            "reasoning": [],
            "auto_executable": []
        }
        
        active_triggers = context["triggers_detected"]
        
        # Analyze each tool against current triggers
        for tool_name, tool_config in self.tool_decision_matrix.items():
            tool_triggers = tool_config["triggers"]
            matching_triggers = set(active_triggers) & set(tool_triggers)
            
            if matching_triggers:
                priority = tool_config["priority"]
                auto_exec = tool_config["auto_execute"]
                
                tool_decision = {
                    "tool": tool_name,
                    "priority": priority,
                    "matching_triggers": list(matching_triggers),
                    "auto_executable": auto_exec
                }
                
                decisions["recommended_tools"].append(tool_decision)
                
                if auto_exec and priority in ["critical", "high"]:
                    decisions["immediate_actions"].append(tool_name)
                    decisions["auto_executable"].append(tool_name)
                elif priority == "medium":
                    decisions["scheduled_actions"].append(tool_name)
                
                reasoning = f"{tool_name}: {', '.join(matching_triggers)}"
                decisions["reasoning"].append(reasoning)
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        decisions["recommended_tools"].sort(
            key=lambda x: priority_order.get(x["priority"], 999)
        )
        
        return decisions
    
    def execute_intelligent_workflow(self):
        """🚀 Execute complete intelligent workflow."""
        print("🧠 VORTA INTELLIGENCE ENGINE - AUTOMATIC TOOL DECISION")
        print("=" * 65)
        
        # Detect current context
        print("🔍 Analyzing current development context...")
        context = self.detect_current_context()
        
        # Make intelligent decisions
        print("🤖 Making intelligent tool decisions...")
        decisions = self.make_intelligent_decisions(context)
        
        # Update memory
        memory = self.load_memory()
        memory["auto_decisions"].append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "decisions": decisions
        })
        self.save_memory(memory)
        
        # Print intelligence report
        self._print_intelligence_report(context, decisions)
        
        # Execute immediate actions
        if decisions["immediate_actions"]:
            print("\n🚀 EXECUTING IMMEDIATE ACTIONS:")
            for tool in decisions["immediate_actions"]:
                print(f"  → Auto-executing: {tool}")
                self._execute_tool(tool)
        
        return decisions
    
    def _print_intelligence_report(self, context: Dict[str, Any], decisions: Dict[str, Any]):
        """Print comprehensive intelligence report."""
        print("\n📊 CONTEXT ANALYSIS:")
        
        # System state
        system = context["system_state"]
        print(f"  🐳 Docker: {system['docker_containers']} containers")
        
        # Service health
        services = context["service_health"]
        healthy_services = sum(1 for h in services.values() if h)
        print(f"  ⚡ Services: {healthy_services}/{len(services)} healthy")
        
        # Development activity
        dev = context["development_activity"]
        print(f"  💻 Development: {'Active' if dev['recent_changes'] else 'Stable'}")
        
        # Active triggers
        triggers = context["triggers_detected"]
        print(f"  🎯 Triggers: {len(triggers)} detected")
        if triggers:
            for trigger in triggers[:3]:  # Show first 3
                print(f"     • {trigger.replace('_', ' ').title()}")
        
        print("\n🧠 INTELLIGENT DECISIONS:")
        
        if decisions["recommended_tools"]:
            for tool_info in decisions["recommended_tools"]:
                tool = tool_info["tool"]
                priority = tool_info["priority"].upper()
                auto = "🤖 AUTO" if tool_info["auto_executable"] else "👤 MANUAL"
                
                print(f"  {priority:8} | {auto} | {tool}")
                
                # Show reasoning
                for trigger in tool_info["matching_triggers"]:
                    print(f"           └─ Trigger: {trigger.replace('_', ' ')}")
        
        print("\n💡 EXECUTION PLAN:")
        if decisions["immediate_actions"]:
            print(f"  🚀 Immediate: {', '.join(decisions['immediate_actions'])}")
        if decisions["scheduled_actions"]:
            print(f"  ⏰ Scheduled: {', '.join(decisions['scheduled_actions'])}")
        if not decisions["recommended_tools"]:
            print("  ✅ No tools needed - system optimal")
    
    def _execute_tool(self, tool_name: str):
        """Execute a specific tool."""
        try:
            if tool_name == "smart_startup.py":
                subprocess.run(['python', 'smart_startup.py', '--status-only'], timeout=30)
            elif tool_name == "quick_check.py":
                subprocess.run(['python', 'quick_check.py'], timeout=15)
            # Other tools would be executed here
            print(f"    ✅ {tool_name} completed")
        except Exception as e:
            print(f"    ❌ {tool_name} failed: {e}")


def main():
    """🧠 Main Intelligence Engine Entry Point."""
    print("🧠 VORTA INTELLIGENCE ENGINE - MEMORY INJECTION ACTIVE")
    print("🔄 Permanent decision rules loaded into memory...")
    
    engine = VortaIntelligenceEngine()
    
    # Execute intelligent workflow
    decisions = engine.execute_intelligent_workflow()
    
    print("\n" + "=" * 65)
    print("🎯 INTELLIGENCE ENGINE: Decision matrix permanently injected!")
    print(f"💾 {len(decisions['recommended_tools'])} intelligent decisions made")
    print("🤖 Automatic tool selection: ACTIVE")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Intelligence engine interrupted")
    except Exception as e:
        print(f"\n❌ Intelligence engine error: {e}")
