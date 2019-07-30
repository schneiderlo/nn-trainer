import os

from setuptools import setup, find_packages

runtime_dependencies = [
  'Pillow>=6.0.0',
  'tensorflow==2.0.0-beta1',
  'pandas',
  'tqdm'
]

build_dependencies = [
  'wheel',
]

dev_dependencies = build_dependencies + [
  'pylint',
  'flake8',
  'flake8-docstrings',
  'pytest',
  'pytest-cov',
  'pytest-html',
  'pytest-xdist',
  'coverage',
]

setup(
  name='nn-trainer',
  version='0.1.0',
  author='schneiderlo',
  description='Provide a easy trainer.',
  long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
  license=open(os.path.join(os.path.dirname(__file__), 'LICENSE')).read(),
  url='https://github.com/schneiderlo/nn-trainer',
  packages=find_packages(exclude=['tests*']),
  # Only used when building binary packages (python setup.py bdist).
  include_package_data=True,
  setup_requires=build_dependencies,
  install_requires=runtime_dependencies,
  extras_require={
    'dev': dev_dependencies
  },
)
