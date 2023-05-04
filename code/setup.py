"""Setup."""
import ast
import re

from setuptools import find_packages, setup

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('rec/__init__.py', 'rb') as f:
    version = str(
        ast.literal_eval(_version_re.search(f.read().decode('utf-8')).group(1))
    )

setup(
    name='rec',
    version=version,
    description='Rec',
    url='',
    author='Nicolas Hörmann',
    author_email='',
    packages=find_packages(),
    dependency_links=[],
    setup_requires=[],
    tests_require=[],
    extras_require={'dev': []},
    entry_points={
        'console_scripts': [
            'predict = rec.predict.__main__:run',
        ]
    },
)