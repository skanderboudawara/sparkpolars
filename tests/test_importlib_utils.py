import re

import pytest

from src.sparkpolars._importlib_utils import check_version_and_module


def test_importlib_utils():

    check_version_and_module("pip", "1.0.0")
    with pytest.raises(ImportError, match=re.escape("Module package_not_existing is not installed")):
        check_version_and_module("package_not_existing", "1.0.0")
    with pytest.raises(ImportError, match=re.escape("Module pip version 99999.0.0 or higher is required")):
        check_version_and_module("pip", "99999.0.0")
