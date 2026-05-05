from setuptools import setup, find_packages

setup(
    name="ml-platform-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["requests>=2.28"],
    python_requires=">=3.9",
    description="Python SDK for ML Platform - push data and trigger analysis",
    author="ML Platform",
)
