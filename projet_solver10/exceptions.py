"""Public exception hierarchy for SolverV10.

Each exception identifies a distinct failure boundary: invalid external model
contracts, unsupported capabilities, unavailable experimental features, or
failed iterative convergence. Callers can catch ``SolverV10Error`` for a
single package-level error boundary.
"""


class SolverV10Error(Exception):
    """Base exception raised by SolverV10."""


class ModelContractError(SolverV10Error):
    """Raised when an external model violates the expected contract."""


class CapabilityError(SolverV10Error):
    """Raised when a requested capability is unsupported."""


class SectorError(SolverV10Error):
    """Raised for an invalid sector or sector transition."""


class ConvergenceError(SolverV10Error):
    """Raised when an iterative algorithm misses its target tolerance."""
