import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impetuous-gfa",
    version="0.1.6",
    author="Richard Tjörnhammar",
    author_email="richard.tjornhammar@gmail.com",
    description="Impetuous Group Factor Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richardtjornhammar/impetuous",
    package_dir = {'impetuous':'src'},
    packages=setuptools.find_packages('src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
