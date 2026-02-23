"""CLI entry point for the semantic-claw-router."""

from __future__ import annotations

import argparse
import signal
import sys

from .config import RouterConfig
from .server import create_server


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Claw Router — Intelligent LLM request routing"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override listen host",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Override listen port",
    )
    args = parser.parse_args()

    # Load config
    try:
        config = RouterConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    # Create and start server
    server, router = create_server(config)

    def shutdown(signum, frame):
        print("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        print(f"Semantic Claw Router listening on {config.host}:{config.port}")
        print(f"Models: {', '.join(m.name for m in config.models)}")
        print(f"Fast-path classifier: {'enabled' if config.fast_path.enabled else 'disabled'}")
        print(f"Session pinning: {'enabled' if config.session.enabled else 'disabled'}")
        print(f"Dedup: {'enabled' if config.dedup.enabled else 'disabled'}")
        print(f"Compression: {'enabled' if config.compression.enabled else 'disabled'}")
        print("---")
        server.serve_forever()
    finally:
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(router.close())
        loop.close()


if __name__ == "__main__":
    main()
