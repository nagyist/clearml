"""
ClearML Inc
https://github.com/clearml/clearml
"""

import os.path
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import codecs


def read_text(filepath):
    with codecs.open(filepath, "r", encoding="utf-8") as f:
        return f.read()


here = os.path.dirname(__file__)
# Get the long description from the README file
long_description = read_text(os.path.join(here, 'README.md'))

# fix github, dark logo hack.
long_description = long_description.replace(
    """<img align="center" src="docs/clearml-logo.svg#gh-light-mode-only" alt="Clear|ML"><img align="center" src="docs/clearml-logo-dark.svg#gh-dark-mode-only" alt="Clear|ML">""",  # noqa
    """<a href="https://clear.ml"><img src="https://raw.githubusercontent.com/clearml/clearml/refs/heads/master/docs/clearml-logo.svg" width="250px"></a>""",  # noqa
)


def read_version_string(version_file):
    for line in read_text(version_file).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = read_version_string("clearml/version.py")

requirements = read_text(os.path.join(here, 'requirements.txt')).splitlines()

setup(
    name='clearml',
    version=version,
    description='ClearML - Auto-Magical Experiment Manager, Version Control, and MLOps for AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # The project's main homepage.
    url='https://github.com/clearml/clearml',
    author='ClearML',
    author_email='support@clear.ml',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Version Control',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords='clearml trains development machine deep learning version control machine-learning machinelearning '
             'deeplearning deep-learning experiment-manager experimentmanager',
    packages=find_packages(exclude=['contrib', 'docs', 'data', 'examples', 'tests']),
    install_requires=requirements,
    extras_require={
        's3': [
            'boto3>=1.9',
        ],
        'azure': [
            'azure-storage-blob>=12.0.0',
        ],
        'gs': [
            'google-cloud-storage>=1.13.2',
        ],
        'router': [
            'fastapi>=0.115.2',
            'uvicorn>=0.31.1',
            'httpx>=0.27.2'
        ]
    },
    package_data={
        "clearml": [
            "config/default/*.conf",
            "backend_api/config/default/*.conf",
            "py.typed",
        ]
    },
    include_package_data=True,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'clearml-init = clearml.cli.config.__main__:main',
            'clearml-data = clearml.cli.data.__main__:main',
            'clearml-task = clearml.cli.task.__main__:main',
            'clearml-param-search = clearml.cli.hpo.__main__:main',
            'clearml-debug = clearml.cli.debug.__main__:main'
        ],
    },
)
