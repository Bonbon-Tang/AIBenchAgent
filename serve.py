#!/usr/bin/env python3

import argparse
import os

import uvicorn

from aibench_service.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AIBenchAgent execution service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", default="./var/jobs")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent evaluation jobs")
    args = parser.parse_args()

    os.environ["AIBENCH_DATA_DIR"] = args.data_dir
    os.environ["AIBENCH_MAX_WORKERS"] = str(args.workers)
    app = create_app(data_dir=args.data_dir, max_workers=args.workers)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
