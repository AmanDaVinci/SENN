import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="senn",
    version="0.0.1",
    author="Ivan Bardarov, Aman Hussain, Christoph Hoenes, Omar Elbaghdadi",
    author_email="ivan.bardarov@student.uva.nl, aman.hussain@student.uva.nl, christoph.hoenes@gmail.com, omarelb@gmail.com",
    description="A review with extensions on SENN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmanDaVinci/SENN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
