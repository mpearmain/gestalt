from setuptools import setup, find_packages

setup(
    name='gestalt',
    version='0.1.1',
    url='https://github.com/mpearmain/gestalt',
    packages=find_packages(),
    description='Data science pipeline automation',
    install_requires=[
        "numpy >= 1.12",
        "pandas >= 0.19.0",
        "scipy >= 0.18.1",
        "scikit-learn >= 0.18.0"
    ],
)