from setuptools import find_packages, setup

setup(
    name='chronos-dysts',
    description='Chronos Fine-Tuned on Dynamical Systems',
    packages=find_packages(exclude=('scripts', 'wandb', 'tests', 'notebooks')),
)