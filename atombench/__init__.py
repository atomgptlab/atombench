"""atombench: compute metrics and generate plots for crystal structure reconstruction benchmarks.

Also supports validating and submitting reconstruction benchmarks directly to the
JARVIS-Leaderboard; see :func:`atombench.submit`.
"""

from atombench.submit import submit
from atombench._leaderboard import SubmissionError, validate_submission

__all__ = ["submit", "validate_submission", "SubmissionError"]
