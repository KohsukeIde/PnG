# Debug utilities for optimization diagnostics

from .optimization_diagnostics import (
    save_optimization_diagnostics_unified,
    save_optimization_diagnostics_SE3,
    mean_parallax_angle,
    debug_transport_matrix_issues,
    collect_sinkhorn_debug_data,
    log_optimization_progress,
    save_transport_snapshot
)