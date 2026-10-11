"""Resource scheduling only; does not alter experimental settings or budgets."""
import math
import time


class SessionBudgetReached(Exception):
    pass


class SessionBudget:
    def __init__(self, minutes, session_start_epoch=None):
        minutes = float(minutes)
        if not math.isfinite(minutes) or minutes <= 0:
            raise ValueError("Choose a finite positive GPU-session limit in minutes")
        consumed = max(0., time.time() - session_start_epoch) if session_start_epoch is not None else 0.
        self.deadline = time.monotonic() + max(0., minutes * 60 - consumed)
        self.minutes = minutes

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SessionBudgetReached("Configured GPU-session time reached; completed work is retained for resumption")


class UnlimitedTestBudget:
    """For tiny CPU tests only; production requires a user-specified limit."""
    def check(self):
        pass
