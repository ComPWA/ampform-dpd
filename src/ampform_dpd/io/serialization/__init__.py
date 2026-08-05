"""Amplitude model serialization

See :doc:`/serialization` for more information and
https://rub-ep1.github.io/amplitude-serialization for more information.

.. warning::
    This module is in preview, see https://github.com/ComPWA/ampform-dpd/issues/133 for
    updates.
"""

from ampform_dpd.io.serialization.compiler import CompiledWorkspace, compile_workspace
from ampform_dpd.io.serialization.kinematics import formulate_kinematic_map
from ampform_dpd.io.serialization.workspace import Workspace, load_workspace

__all__ = [
    "CompiledWorkspace",
    "Workspace",
    "compile_workspace",
    "formulate_kinematic_map",
    "load_workspace",
]
