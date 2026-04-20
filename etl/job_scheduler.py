"""
Optional daily ETL automation using `schedule`.

Run from project root:
    python -m etl.job_scheduler

Override run time (24h clock, local machine):
    set ETL_SCHEDULE_TIME=09:30
"""

import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.logging_config import configure_logging

configure_logging()

import schedule

from etl.run_pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _run_etl_job():
    logger.info("Scheduled ETL run starting")
    try:
        run_pipeline()
        logger.info("Scheduled ETL run finished")
    except Exception:
        logger.exception("Scheduled ETL run failed")


def main():
    run_at = os.getenv("ETL_SCHEDULE_TIME", "10:00")
    schedule.every().day.at(run_at).do(_run_etl_job)
    logger.info(
        "Scheduler running: daily ETL at %s (override with ETL_SCHEDULE_TIME)",
        run_at,
    )

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
