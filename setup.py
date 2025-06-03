# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

def setup_package():
    setup(
        name="torchfry",
        version="1.0.0",
        author="Robert Bates, Kameron Decker Harris, Jed Christian Pagcaliwagan, Josh Sonnen",
        description="Provide PyTorch layers and networks that utilize the Fastfood algorithm and Random Kitchen Sink algorithms.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
        ],
        python_requires=">=3.9",
        install_requires=["torch", "scipy"],
        packages=find_packages(
            exclude=(
                "build",
                "data",
                "tests",
                "docs"
            )
        )
    )

if __name__ == "__main__":
    setup_package()