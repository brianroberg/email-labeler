"""Launch the eval web UI.

Usage:
    python -m evals.run_web
    python -m evals.run_web --port 8080
"""

import argparse

import uvicorn


def cli():
    parser = argparse.ArgumentParser(description="Launch eval web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    args = parser.parse_args()

    uvicorn.run(
        "evals.web_app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
