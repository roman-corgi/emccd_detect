import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

setuptools.setup(
    name="arcticpy",
    packages=setuptools.find_packages(exclude=['emccd_detect.*', 'emccd_detect']),
    version="1.0",
    description="AlgoRithm for Charge Transfer Inefficiency Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob Kegerreis, Richard Massey, James Nightingale",
    author_email="jacob.kegerreis@durham.ac.uk",
    url="https://github.com/jkeger/arcticpy",
    license="GNU GPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    #python_requires=">=3",
    python_requires="<=3.9",
    install_requires=requirements,
    keywords=["charge transfer inefficiency correction"],
)
