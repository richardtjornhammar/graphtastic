# A Statistical Graph Learning library for Humans

These novel algorithms include but are not limited to:
* A graph construction and graph searching class can be found in src/impetuous/convert.py (GraphNode). It was developed and invented as a faster alternative for hierarchical DAG construction and searching.
* A fast DBSCAN method utilizing [my](https://richardtjornhammar.github.io/) connectivity code as invented during my PhD.
* A NLP pattern matching algorithm useful for sequence alignment clustering
* High dimensional alignment code for aligning models to data.
* An SVD based variant of the Distance Geometry algorithm. For going from relative to absolute coordinates.

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0)

Visit the active code via :
https://github.com/richardtjornhammar/graphtastic

# Pip installation with :
```
pip install graphtastic
```

# Version controlled installation of the Graphtastic library

The Graphtastic library

In order to run these code snippets we recommend that you download the nix package manager. Nix package manager links from Oktober 2020:

https://nixos.org/download.html

```
$ curl -L https://nixos.org/nix/install | sh
```

If you cannot install it using your Wintendo then please consider installing Windows Subsystem for Linux first:

```
https://docs.microsoft.com/en-us/windows/wsl/install-win10
```

In order to run the code in this notebook you must enter a sensible working environment. Don't worry! We have created one for you. It's version controlled against python3.7 (and python3.8) and you can get the file here:

https://github.com/richardtjornhammar/rixcfgs/blob/master/code/environments/impetuous-shell.nix

Since you have installed Nix as well as WSL, or use a Linux (NixOS) or bsd like system, you should be able to execute the following command in a termnial:

```
$ nix-shell impetuous-shell.nix
```

Now you should be able to start your jupyter notebook locally:

```
$ jupyter-notebook graphhaxxor.ipynb
```

and that's it.
