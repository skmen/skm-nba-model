"""
Scheduler Module

Handles time-based scheduling for automated daily predictions.
Supports both APScheduler and native cron jobs.

Author: NBA Prediction Model
Date: 2025
"""

import logging
import os
import sys
from typing import Callable, Optional, Union
from datetime import datetime, time
import signal

from src.utils import setup_logger, PipelineError

logger = setup_logger(__name__)


class SchedulerError(PipelineError):
    """Scheduler-specific errors."""
    pass


class TimeBasedScheduler:
    """
    Simple scheduler using system time checks.
    Lightweight alternative to APScheduler for simple use cases.
    """
    
    def __init__(self, run_time: str = "09:00"):
        """
        Initialize scheduler.
        
        Args:
            run_time: Time to run job in HH:MM format (24-hour, e.g., "09:00")
        """
        self.run_time = run_time
        self.is_running = False
        self.job_func = None
        self.last_run_date = None
        
        # Parse run time
        try:
            self.hour, self.minute = map(int, run_time.split(':'))
        except ValueError:
            raise SchedulerError(f"Invalid time format: {run_time}. Use HH:MM (24-hour)")
    
    def schedule_daily(self, func: Callable) -> None:
        """
        Schedule function to run daily at specified time.
        
        Args:
            func: Callable to execute daily
        """
        self.job_func = func
        logger.info(f"Scheduled daily job at {self.run_time}")
    
    def start(self) -> None:
        """
        Start the scheduler (blocking, runs indefinitely).
        Press Ctrl+C to stop.
        """
        if self.job_func is None:
            raise SchedulerError("No job scheduled. Call schedule_daily() first.")
        
        logger.info("Starting scheduler...")
        self.is_running = True
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Scheduler shutting down...")
            self.is_running = False
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main loop
        while self.is_running:
            now = datetime.now()
            current_time = time(now.hour, now.minute)
            target_time = time(self.hour, self.minute)
            
            # Check if it's time to run
            if current_time == target_time and self.last_run_date != now.date():
                logger.info(f"Running scheduled job at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                try:
                    self.job_func()
                    self.last_run_date = now.date()
                    logger.info("Job completed successfully")
                except Exception as e:
                    logger.error(f"Job failed: {e}", exc_info=True)
            
            # Sleep to avoid busy waiting
            import time as time_module
            time_module.sleep(30)  # Check every 30 seconds


class APSchedulerWrapper:
    """
    Wrapper around APScheduler for more sophisticated scheduling.
    Requires: pip install apscheduler
    """
    
    def __init__(self, run_time: str = "09:00"):
        """
        Initialize APScheduler-based scheduler.
        
        Args:
            run_time: Time to run in HH:MM format
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            raise SchedulerError(
                "APScheduler not installed. Install with: pip install apscheduler"
            )
        
        self.run_time = run_time
        self.scheduler = BackgroundScheduler()
        self.job_id = "daily_prediction_job"
        
        # Parse time
        try:
            self.hour, self.minute = map(int, run_time.split(':'))
        except ValueError:
            raise SchedulerError(f"Invalid time format: {run_time}. Use HH:MM")
        
        logger.info(f"Initialized APScheduler for {run_time}")
    
    def schedule_daily(self, func: Callable) -> None:
        """
        Schedule function to run daily at specified time.
        
        Args:
            func: Callable to execute daily
        """
        try:
            from apscheduler.triggers.cron import CronTrigger
            
            trigger = CronTrigger(hour=self.hour, minute=self.minute)
            
            self.scheduler.add_job(
                func,
                trigger=trigger,
                id=self.job_id,
                name="Daily Prediction Job",
                replace_existing=True
            )
            
            logger.info(f"Scheduled job using APScheduler (daily at {self.run_time})")
            
        except Exception as e:
            raise SchedulerError(f"Failed to schedule job: {e}")
    
    def start(self) -> None:
        """Start the scheduler."""
        try:
            logger.info("Starting APScheduler...")
            self.scheduler.start()
            logger.info("Scheduler is running. Press Ctrl+C to stop.")
            
            # Keep running
            try:
                import time as time_module
                while True:
                    time_module.sleep(1)
            except KeyboardInterrupt:
                logger.info("Scheduler shutting down...")
                self.scheduler.shutdown()
                
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise SchedulerError(f"Scheduler failed: {e}")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")


def get_scheduler(
    run_time: str = "09:00",
    use_apscheduler: bool = False
) -> Union['TimeBasedScheduler', 'APSchedulerWrapper']:
    """
    Get appropriate scheduler instance.
    
    Args:
        run_time: Time to run in HH:MM format
        use_apscheduler: Use APScheduler (if True) or simple scheduler (if False)
        
    Returns:
        Scheduler instance
    """
    if use_apscheduler:
        try:
            return APSchedulerWrapper(run_time)
        except SchedulerError as e:
            logger.warning(f"Could not use APScheduler: {e}. Falling back to simple scheduler.")
            return TimeBasedScheduler(run_time)
    else:
        return TimeBasedScheduler(run_time)


# Example cron setup for system cron (macOS/Linux)
CRON_TEMPLATE = """
# NBA Daily Prediction Job
# Add to crontab with: crontab -e
# Then add this line (runs at 9:00 AM daily):

0 9 * * * cd {project_dir} && {python_exe} {script_path} >> {log_file} 2>&1
"""


def generate_cron_entry(
    project_dir: str,
    script_path: str,
    run_time: str = "09:00",
    log_file: Optional[str] = None
) -> str:
    """
    Generate cron entry for system cron job.
    
    Args:
        project_dir: Path to project directory
        script_path: Path to automation script
        run_time: Time in HH:MM format
        log_file: Path to log file (optional)
        
    Returns:
        Cron entry string
    """
    python_exe = sys.executable
    
    if log_file is None:
        log_file = os.path.join(project_dir, "logs/automation.log")
    
    # Parse time
    hour, minute = map(int, run_time.split(':'))
    
    cron_entry = f"{minute} {hour} * * * cd {project_dir} && {python_exe} {script_path} >> {log_file} 2>&1"
    
    return cron_entry


def print_cron_setup(
    project_dir: str,
    script_path: str,
    run_time: str = "09:00",
    log_file: Optional[str] = None
) -> None:
    """
    Print instructions for setting up system cron job.
    
    Args:
        project_dir: Path to project directory
        script_path: Path to automation script
        run_time: Time in HH:MM format
        log_file: Path to log file
    """
    cron_entry = generate_cron_entry(project_dir, script_path, run_time, log_file)
    
    print("\n" + "="*70)
    print("📅 SYSTEM CRON SETUP INSTRUCTIONS")
    print("="*70)
    print(f"\nTo run predictions daily at {run_time}, add this line to your crontab:\n")
    print(f"  {cron_entry}\n")
    print("How to add:")
    print("  1. Open crontab editor:  crontab -e")
    print("  2. Paste the line above")
    print("  3. Save and exit (Ctrl+X, then Y in nano editor)")
    print("\nTo view scheduled cron jobs:")
    print("  crontab -l")
    print("\nTo view logs:")
    print(f"  tail -f {log_file}")
    print("="*70 + "\n")
