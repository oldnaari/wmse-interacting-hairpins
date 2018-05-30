import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wmse_cbs",
    version="0.0.1",
    author="oldnaari",
    author_email="daniil.hayrapetyan@gmail.com",
    description="WMSE Complex Beta Structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oldnaari/wmse-interacting-hairpins.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)