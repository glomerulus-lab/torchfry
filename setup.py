# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

def setup_package():
    setup(
        name="fastfood_torch",
        version="0.0.1dev0",
        description="Provide PyTorch layers and networks that utilize the Fastfood algorithm and Random Kitchen Sink algorithms.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=["torch"],
        packages=find_packages()
    )

if __name__ == "__main__":
    setup_package()