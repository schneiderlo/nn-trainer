import os
from shutil import rmtree

from nn_trainer import ROOT_DIR


def get_project_root() -> str:
  """Return the absolute path to the project directory."""
  return ROOT_DIR


def get_test_data(*args) -> str:
  """Return the absolute path to data.

  Args:
    args: List of string.
  """
  return os.path.join(
    get_project_root(),
    'tests',
    'data',
    *args
  )


def get_test_directory(request, clear: bool = True):
  """

  Args:
    request:
    clear:

  Return:
  """
  dir_name = request.node.name
  directory_path = os.path.join(
    get_project_root(),
    'build',
    'tests',
    'workspace',
    dir_name
  )
  if os.path.exists(directory_path) and os.path.isdir(directory_path) and clear:
    rmtree(directory_path)
  os.makedirs(directory_path, exist_ok=True)
  return directory_path
