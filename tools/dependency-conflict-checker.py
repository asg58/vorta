#!/usr/bin/env python3
"""
🔍 VORTA ULTRA Dependency Conflict Checker
Pre-Build Validation Tool to Catch All Package Conflicts

EXACTLY what you suggested - test BEFORE building!
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List


class DependencyConflictChecker:
    """Ultra Professional Dependency Conflict Detection"""
    
    def __init__(self, requirements_file: Path):
        self.requirements_file = requirements_file
        self.conflicts_found = []
        self.warnings = []
        
    def print_status(self, message: str, status: str = "INFO"):
        """Professional status output"""
        colors = {
            "INFO": "\033[96m",     # Cyan
            "SUCCESS": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",    # Red
            "RESET": "\033[0m"      # Reset
        }
        
        timestamp = time.strftime("%H:%M:%S")
        color = colors.get(status, colors["INFO"])
        reset = colors["RESET"]
        
        print(f"{color}[{timestamp}] {status}: {message}{reset}")
    
    def test_dependency_resolution(self) -> bool:
        """Test full dependency resolution WITHOUT installation"""
        self.print_status("🔍 Testing dependency resolution (dry-run)...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create virtual environment for testing
                venv_path = Path(temp_dir) / "test_venv"
                result = subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    error_msg = f"Failed to create test environment: {result.stderr}"
                    self.print_status(error_msg, "ERROR")
                    return False
                
                # Get pip executable path
                if sys.platform == "win32":
                    pip_exe = venv_path / "Scripts" / "pip.exe"
                    python_exe = venv_path / "Scripts" / "python.exe"
                else:
                    pip_exe = venv_path / "bin" / "pip"
                    python_exe = venv_path / "bin" / "python"
                
                # Upgrade pip first
                subprocess.run([
                    str(python_exe), "-m", "pip", "install", "--upgrade", "pip"
                ], capture_output=True, timeout=30)
                
                # Test dependency resolution with simple install check
                self.print_status("Running pip dependency resolver...")
                
                cmd = [
                    str(pip_exe), "install", 
                    "--dry-run", 
                    "-r", str(self.requirements_file)
                ]
                
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                
                if result.returncode != 0:
                    self.print_status("❌ Dependency conflicts detected!", "ERROR")
                    self.parse_pip_errors(result.stderr)
                    return False
                else:
                    self.print_status("✅ All dependencies compatible!", "SUCCESS")
                    return True
                    
            except subprocess.TimeoutExpired:
                self.print_status("Dependency resolution timed out", "ERROR")
                return False
            except Exception as e:
                self.print_status(f"Test failed: {e}", "ERROR")
                return False
    
    def parse_pip_errors(self, stderr_output: str):
        """Parse pip error output for specific conflicts"""
        lines = stderr_output.split('\n')
        
        current_conflict = None
        in_conflict_section = False
        
        for line in lines:
            line = line.strip()
            
            # Detect conflict start
            if "Cannot install" in line and "because these package versions have conflicting dependencies" in line:
                current_conflict = {"packages": line, "details": []}
                in_conflict_section = True
                self.print_status(f"🚨 CONFLICT: {line}", "ERROR")
                
            elif in_conflict_section and line.startswith("The conflict is caused by:"):
                continue
                
            elif in_conflict_section and ("depends on" in line or "requested" in line):
                current_conflict["details"].append(line)
                self.print_status(f"   - {line}", "WARNING")
                
            elif in_conflict_section and line.startswith("To fix this"):
                if current_conflict:
                    self.conflicts_found.append(current_conflict)
                in_conflict_section = False
                current_conflict = None
    
    def get_package_versions_from_pypi(self, package_name: str) -> List[str]:
        """Get available versions from PyPI (for suggestions)"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "index", "versions", package_name
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse output to extract versions
                lines = result.stdout.split('\n')
                versions = []
                for line in lines:
                    if "Available versions:" in line:
                        version_part = line.split("Available versions:")[1].strip()
                        versions = [v.strip() for v in version_part.split(',')]
                        break
                return versions[:5]  # Return top 5 versions
            
        except Exception:
            pass
        
        return []
    
    def suggest_fixes(self):
        """Suggest specific fixes for detected conflicts"""
        if not self.conflicts_found:
            return
            
        self.print_status("🛠️  SUGGESTED FIXES:", "INFO")
        
        for i, conflict in enumerate(self.conflicts_found, 1):
            self.print_status(f"\nConflict #{i}:", "WARNING")
            
            for detail in conflict["details"]:
                if "depends on" in detail:
                    # Extract package and version requirement
                    parts = detail.split(" depends on ")
                    if len(parts) == 2:
                        dependent = parts[0].strip()
                        requirement = parts[1].strip()
                        
                        self.print_status(f"   📦 Consider updating: {requirement}", "INFO")
    
    def run_comprehensive_check(self) -> bool:
        """Run complete dependency validation"""
        self.print_status("🚀 VORTA ULTRA Dependency Conflict Checker", "INFO")
        self.print_status(f"📄 Checking: {self.requirements_file}", "INFO")
        
        # Step 1: Validate requirements file exists
        if not self.requirements_file.exists():
            self.print_status("Requirements file not found!", "ERROR")
            return False
        
        # Step 2: Test dependency resolution
        if not self.test_dependency_resolution():
            self.print_status("\n❌ DEPENDENCY CONFLICTS DETECTED!", "ERROR")
            self.suggest_fixes()
            return False
        
        self.print_status("\n✅ ALL DEPENDENCIES COMPATIBLE - BUILD SAFE!", "SUCCESS")
        return True


def main():
    """Main execution"""
    if len(sys.argv) != 2:
        print("Usage: python dependency-conflict-checker.py <requirements.txt>")
        sys.exit(1)
    
    requirements_file = Path(sys.argv[1])
    checker = DependencyConflictChecker(requirements_file)
    
    success = checker.run_comprehensive_check()
    
    if not success:
        print("\n🛑 FIX CONFLICTS BEFORE BUILDING!")
        sys.exit(1)
    else:
        print("\n🎯 READY FOR BUILD!")
        sys.exit(0)


if __name__ == "__main__":
    main()
