# Welcome to AmpForm-DPD!

```{title} Welcome

```

This Python package is a (temporary) extension of [AmpForm](https://ampform.rtfd.io) and provides a symbolic implementation of Dalitz-plot decomposition ([10.1103/PhysRevD.101.034033](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033)) with [SymPy](https://www.sympy.org/en/index.html). It has been extracted from the [ComPWA/polarimetry](https://github.com/ComPWA/polarimetry) repository, which is not yet public.

## Installation

The fastest way of installing this package is through PyPI:

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform-dpd@stable
```

This installs the latest version that you can find on the [`stable`](https://github.com/ComPWA/ampform-dpd/tree/stable) branch. You can substitute `stable` in the above command with `main` or any of the [tags](https://github.com/ComPWA/ampform-dpd/tags) listed under the [releases](https://github.com/ComPWA/ampform-dpd/releases). In a similar way, you can list `ampform-dpd` as a dependency of your application in either `setup.cfg` or a `requirements.txt` file as:

```text
ampform-dpd @ git+https://github.com/ComPWA/ampform-dpd@main
```

However, we highly recommend using the more dynamic, {ref}`'editable installation' <compwa-org:develop:Editable installation>` instead. This goes as follows:

1. Get the source code (see [the Pro Git Book](https://git-scm.com/book/en/v2)):

   ```shell
   git clone https://github.com/ComPWA/ampform-dpd.git
   cd ampform-dpd
   ```

2. **[Recommended]** Create a virtual environment (see {ref}`here <compwa-org:develop:Virtual environment>` or the tip below).

3. Install the project in {ref}`'editable installation' <compwa-org:develop:Editable installation>` with {ref}`additional dependencies <compwa-org:develop:Optional dependencies>` for the developer:

   ```shell
   python3 -m pip install -e .[dev]
   ```

That's all! Have a look at the {doc}`/index` page to try out the package, and see {doc}`compwa-org:develop` for tips on how to work with this 'editable' developer setup!

:::{tip}

It's easiest to install the project in a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment. In that case, to install in editable mode, just run:

```shell
conda env create
conda activate ampform-dpd
```

This way of installing is also safer, because it {ref}`pins all dependencies <compwa-org:develop:Pinning dependency versions>`. Note you can also pin dependencies with `pip`, by running:

```shell
python3 -m pip install -e .[dev] -c .constraints/py3.x.txt
```

where you should replace the `3.x` with the version of Python you want to use.

:::

<!-- cspell:ignore pkpi -->

## Examples

```{toctree}
---
maxdepth: 1
---
lc2pkpi
jpsi2ksp
```

```{toctree}
---
hidden:
---
API <api/ampform_dpd>
```
