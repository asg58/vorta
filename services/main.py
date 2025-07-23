# services/main.py

import argparse
import asyncio
import os
import uvicorn

def main():
    """
    Main entry point to run VORTA backend services.
    
    This script allows starting individual services or all services at once.
    It uses uvicorn to run FastAPI applications.
    """
    parser = argparse.ArgumentParser(description="VORTA AGI Backend Service Runner")
    parser.add_argument(
        "service",
        nargs="?",
        default="all",
        choices=["orchestrator", "vector_store_mock", "all"],
        help="The service to run. 'vector_store_mock' runs a mock server for the vector store.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the service to."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the service on. Defaults are used if not specified."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reloading for development."
    )

    args = parser.parse_args()

    service_configs = {
        "orchestrator": {
            "app": "services.orchestrator.service_coordinator:app",
            "port": 8000,
            "host": "127.0.0.1",
        },
        "vector_store_mock": {
            "app": "services.vector_store.mock_server:app", # We will create this mock server
            "port": 8001,
            "host": "127.0.0.1",
        },
    }

    if args.service == "all":
        print("Starting all services...")
        # This part is complex to run in a single script due to asyncio event loops.
        # For production, Docker Compose or Kubernetes is the right tool.
        # For development, it's often better to run each service in a separate terminal.
        print("Running all services in a single process is not recommended for development.")
        print("Please run each service in a separate terminal:")
        print("python -m services.main orchestrator --reload")
        print("python -m services.main vector_store_mock --reload")
        return

    if args.service in service_configs:
        config = service_configs[args.service]
        port = args.port if args.port is not None else config["port"]
        host = args.host
        
        print(f"Starting {args.service} service on http://{host}:{port}")
        
        uvicorn.run(
            config["app"],
            host=host,
            port=port,
            reload=args.reload,
            log_level="info",
        )
    else:
        print(f"Error: Service '{args.service}' not found.")
        parser.print_help()

if __name__ == "__main__":
    # To run this, you would use the command line, for example:
    # python -m services.main orchestrator
    main()
