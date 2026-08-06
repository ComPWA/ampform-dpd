"""Amplitude model serialization

See :doc:`/serialization` for more information and
https://rub-ep1.github.io/amplitude-serialization for more information.

.. warning::
    This module is in preview, see https://github.com/ComPWA/ampform-dpd/issues/133 for
    updates.
"""

from ampform_dpd.io.serialization.workspace import Workspace, load_workspace

__all__ = ["Workspace", "load_workspace"]
