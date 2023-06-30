import setuptools


# Put your python dependencies in here
requirements = ["adapter-transformers==3.0.1",
                "numpy",
                "torch==1.13.1",
                "transformers==4.21.1"
                ]


setuptools.setup(
    name="specter2_0",
    version="1.0.1",
    description="Embeddings for scientifiic papers",
    url="https://github.com/allenai/SPECTER2_0",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={
        "timo": [
            "build",
            "pydantic",
            "pytest"
        ]
    },
    python_requires="~= 3.8"
)
