import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "graphtastic",
    version = "0.10.2",
    author = "Richard Tjörnhammar",
    author_email = "richard.tjornhammar@gmail.com",
    description = "Graphtastic, a Statistical Graph Learning library for Humans",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/richardtjornhammar/graphtastic",
    packages = setuptools.find_packages('src'),
    package_dir = {'graphtastic':'src/graphtastic','convert':'src/convert','clustering':'src/clustering','fit':'src/fit'},
    classifiers = [
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
