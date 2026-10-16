"""Editable installation of the FiBA research source."""
from setuptools import find_packages, setup

setup(
    name='fiba', version='2.0.0',
    description='On the Effectiveness of Invisible Backdoor Attacks in Federated Learning',
    python_requires='>=3.11,<3.14', license='GPL-3.0-only',
    packages=find_packages('src'), package_dir={'':'src'},
    install_requires=['torch==2.8.0','torchvision==0.23.0','flwr[simulation]==1.40.0',
                      'numpy>=2,<3','torchcam==0.5.0','lpips==0.1.4'],
    entry_points={'console_scripts':['fiba=fiba.experiment:main']},
)
