"""Witness Companion coordinator: hardware-free core.

Importing this package touches no device, no network, and no file outside
paths the caller passes in. Real device adapters are staged, not written,
and must be lazy-imported behind an explicit --armed switch
(see companion/adapters/real/README.md).

Stdlib only. Python 3.10 compatible.
"""

__all__ = ["schemas", "guard", "persist", "coordinator"]
