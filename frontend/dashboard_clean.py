"""
VORTA Enterprise Dashboard - Production Entry Point

Author: High-Grade Development Team
Version: 2.0.0-enterprise
License: MIT
"""

import os
import sys

# Configure module path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)


def main():
    """Enterprise application entry point"""
    try:
        print("🚀 VORTA Enterprise Platform - Loading...")
        print("🏭 Initializing production-grade architecture...")
        
        # Import and launch the enterprise dashboard
        from vorta_enterprise_dashboard import run_enterprise_dashboard
        
        print("✅ Enterprise modules loaded successfully!")
        print("🎯 Launching VORTA Enterprise Dashboard...")
        
        # Run the enterprise application
        run_enterprise_dashboard()
        
    except ImportError as e:
        print(f"""
🚨 ENTERPRISE MODULE IMPORT ERROR

The required enterprise modules could not be imported.
Error: {e}

📋 SOLUTION - Install Required Dependencies:

1. Open terminal in the frontend directory
2. Run the following commands:

   pip install streamlit requests numpy pandas plotly psutil
   pip install streamlit-autorefresh

3. Restart the application

🔧 Alternative - Auto Install:
   python -m pip install -r ../requirements.txt
        """)
        
    except Exception as e:
        print(f"""
🚨 APPLICATION STARTUP ERROR

An unexpected error occurred during startup.
Error: {e}

📋 TROUBLESHOOTING STEPS:

1. Verify all dependencies are installed
2. Check Python environment configuration
3. Ensure all module files are present:
   - vorta_enterprise_dashboard.py
   - api_client/enterprise_client.py
   - components/enterprise_ai_interface.py
   - ui_themes/enterprise_theme.py

4. Contact development team if issue persists
        """)


if __name__ == "__main__":
    main()
