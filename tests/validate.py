"""Compatibility shim — CI discovery redirect (issue #523).

The tests formerly in this monolith have been split into per-module files under tests/.
This file exists ONLY because the Docker image's baked ci_entrypoint.sh still references
``tests/validate.py`` as the pytest target. Once the image is rebuilt with the updated
ci_entrypoint.sh (which uses ``tests/`` as the target directory), this file can be deleted.

The wildcard imports below re-export every test function so that::

    pytest tests/validate.py -m smoke

still discovers all smoke/integration tests.
"""
# ruff: noqa: F401, F403 — wildcard imports intentional for pytest collection

from .test_calibration import *  # noqa
from .test_composition import *  # noqa
from .test_config_validate import *  # noqa
from .test_evolution import *  # noqa
from .test_fgong import *  # noqa
from .test_golden_master import *  # noqa
from .test_grad_timing import *  # noqa
from .test_gradient_policy import *  # noqa
from .test_inference import *  # noqa
from .test_interp_utils import *  # noqa
from .test_mesh import *  # noqa
from .test_microphysics import *  # noqa
from .test_oscillations import *  # noqa
from .test_solver import *  # noqa
from .test_solver_units import *  # noqa
from .test_timing_instrumentation import *  # noqa
from .test_transport_validate import *  # noqa
