import os
import numba as nb
import logging

# Use of numba jitting controlled by this environment variable.
# Enabled by default
NUMBA_ENABLED = "DISABLE_NUMBA" not in os.environ or os.environ["DISABLE_NUMBA"] in {"0", "false", "False", "FALSE"}


if NUMBA_ENABLED:
    jitclass = nb.experimental.jitclass
    njit = nb.njit
    jit = nb.jit
else:
    logger = logging.getLogger(__name__)
    logger.warn("Numba jit classes and functions are disabled")

    def jitclass(spec=None):
        def decorator(cls):
            return cls
        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
