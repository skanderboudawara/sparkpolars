from importlib.metadata import version
from importlib.util import find_spec


def check_version_and_module(module_name: str, minimum_version: str) -> None:
    """
    This function checks if a module is installed and
        if its version is greater than or equal to the minimum version.

    :param module_name: The name of the module to check

    :param minimum_version: The minimum version required

    :return: None
    """
    spec = find_spec(module_name)
    if spec is None:
        msg = f"Module {module_name} is not installed."
        raise ImportError(msg)

    if version(module_name) <= minimum_version:
        msg = f"Module {module_name} version {minimum_version} or higher is required."
        raise ImportError(msg)
