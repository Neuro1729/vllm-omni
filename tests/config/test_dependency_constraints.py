# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_COMMON_REQUIREMENTS = Path(__file__).parents[2] / "requirements" / "common.txt"


def test_common_requirements_keep_hub_kernel_dependencies_compatible():
    requirements = _COMMON_REQUIREMENTS.read_text()

    assert re.search(r"^transformers\s*>=\s*5\.13\.0,\s*<\s*5\.15\s*$", requirements, re.MULTILINE)
    assert re.search(r"^kernels\s*==\s*0\.16\.0\s*$", requirements, re.MULTILINE)
