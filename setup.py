import setuptools


# Put your python dependencies in here
requirements = []


setuptools.setup(
    name="specter2_0",
    version="0.0.1",
    description="<TODO>",
    url="<TODO>",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={
        "timo": [
            "pydantic",
            "pytest"
        ]
    },
    python_requires="~= 3.8"
)
