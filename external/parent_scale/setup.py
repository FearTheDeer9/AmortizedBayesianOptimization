from setuptools import setup, find_packages

setup(
    name="parent_scale",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", 
        "torch",
        "tqdm",
        "scipy",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)