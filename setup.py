#!/usr/bin/env python
# coding=utf-8
# by Jiajia Liu @ Queen's University Belfast
# March 2021

from setuptools import setup, find_packages
import glob
import os


def main():
    setup(
        name="sunslice",
        python_requires='>=3.5.0',
        version="1.0",
        author="Jiajia Liu",
        author_email="j.liu@qub.ac.uk",
        description=("A Python package for creating different types of slices and generating time-distance diagrams from solar 2D observations"),
        license="GPLv3",
        keywords="ASDA",
        url="https://github.com/PyDL/pysun_slice",
        packages=find_packages(where='.', exclude=(), include=('*',)),
        py_modules=get_py_modules(),

        # dependencies
        install_requires=[
            'numpy',
            'astropy',
            'sunpy>=3.0.0',
            'matplotlib',
            'h5py'
        ],

        classifiers=[
            "Development Status :: 1.0 - Release",
            "Topic :: Utilities",
            "License :: OSI Approved :: GNU General Public License (GPL)",
        ],

        zip_safe=False
    )


def get_py_modules():
    py_modules=[]
    for file in glob.glob('*.py'):
        py_modules.append(os.path.splitext(file)[0])

    print(py_modules)
    return py_modules


if __name__ == "__main__":
    main()
