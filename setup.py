
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='acnportal',
    version='0.1',
    author='Zachary Lee',
    author_email="zlee@caltech.edu",
    description="A package of tools for large-scale EV charging research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
     "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'requests',
        'pytz'
    ],

 )