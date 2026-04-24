from setuptools import setup, find_packages
import os

def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mikasa_robo_suite",
    version="0.0.5",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'), 
    author="Egor Cherepanov",
    author_email="cherepanovegor2018@gmail.com",
    description="Gym-like memory-intensive environmtnts for robotic tabletop manipulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CognitiveAISystems/MIKASA-Robo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)