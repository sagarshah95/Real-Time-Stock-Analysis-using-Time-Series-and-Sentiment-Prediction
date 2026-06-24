#!/usr/bin/env python3
"""Run the X → Kafka → trend store pipeline (producer + consumer) from the terminal."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streaming.x_consumer import start_x_consumer, stop_x_consumer
from streaming.x_producer import backfill_keyword, start_x_producer, stop_x_producer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_x_kafka_stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="X tweet Kafka stream for F.A.S.T social trends")
    parser.add_argument("--keyword", default="Amazon", help="Stock keyword or ticker")
    parser.add_argument("--backfill", action="store_true", help="Backfill recent tweets before streaming")
    args = parser.parse_args()

    keyword = args.keyword
    if args.backfill:
        n = backfill_keyword(keyword, on_progress=logger.info)
        logger.info("Backfill complete: %s tweets", n)

    start_x_consumer()
    start_x_producer(keyword)
    logger.info("Pipeline running for %s. Press Ctrl+C to stop.", keyword)

    def _shutdown(signum, frame):
        logger.info("Shutting down…")
        stop_x_producer(keyword)
        stop_x_consumer()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    main()
