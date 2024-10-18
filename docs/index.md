# Welcome to AmpForm-DPD!

```{title} Welcome

```

[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform-dpd)](https://pypi.org/project/ampform-dpd)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/ampform/blob/main)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/ampform/main?urlpath=lab)

This Python package is a (temporary) extension of [AmpForm](https://ampform.rtfd.io) and provides a symbolic implementation of Dalitz-plot decomposition ([10.1103/PhysRevD.101.034033](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033)) with [SymPy](https://www.sympy.org/en/index.html). It has been extracted from the [ComPWA/polarimetry](https://github.com/ComPWA/polarimetry) repository, which is not yet public.

## Installation

The fastest way of installing this package is through PyPI:

```shell
python3 -m pip install ampform-dpd
```

This installs the latest version that you can find on the [`stable`](https://github.com/ComPWA/ampform-dpd/tree/stable) branch. The latest version on the [`main`](https://github.com/ComPWA/ampform/tree/main) branch
can be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform@main
```

You can substitute `stable` in the above command with `main` or any of the [tags](https://github.com/ComPWA/ampform-dpd/tags) listed under the [releases](https://github.com/ComPWA/ampform-dpd/releases). In a similar way, you can list `ampform-dpd` as a dependency of your application in either `setup.cfg` or a `requirements.txt` file as:

```text
ampform-dpd @ git+https://github.com/ComPWA/ampform-dpd@main
```

However, we highly recommend using the more dynamic, {ref}`'editable installation' <compwa:develop:Editable installation>` instead. This goes as follows:

1. Get the source code (see [the Pro Git Book](https://git-scm.com/book/en/v2)):

   ```shell
   git clone https://github.com/ComPWA/ampform-dpd.git
   cd ampform-dpd
   ```

2. **\[Recommended\]** Create a virtual environment (see {ref}`here <compwa:develop:Virtual environment>` or the tip below).

3. Install the project in {ref}`'editable installation' <compwa:develop:Editable installation>` with {ref}`additional dependencies <compwa:develop:Optional dependencies>` for the developer:

   ```shell
   python3 -m pip install -e .[dev]
   ```

That's all! Have a look at the {doc}`/index` page to try out the package, and see {doc}`compwa:develop` for tips on how to work with this 'editable' developer setup!

:::{tip}

It's easiest to install the project in a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment. In that case, to install in editable mode, just run:

```shell
conda env create
conda activate ampform-dpd
```

This way of installing is also safer, because it {ref}`pins all dependencies <compwa:develop:Pinning dependency versions>`. Note you can also pin dependencies with `pip`, by running:

```shell
python3 -m pip install -e .[dev] -c .constraints/py3.x.txt
```

where you should replace the `3.x` with the version of Python you want to use.

:::

<!-- cspell:ignore pkpi -->

## Physics

Dalitz-plot decomposition allows us to separate variables that affect the <font color=RoyalBlue>angular distribution</font> from variables that describe the <font color=Orange>dynamics</font>. It allows rewriting a **transition amplitude**&nbsp;$T$ as

$$
T^{\Lambda}_{\{\lambda\}}(\alpha,\beta,\gamma; \{\sigma\}) = \sum_{\nu}
{\color{RoyalBlue} D_{\Lambda,\nu}^{J}(\alpha,\beta,\gamma)}
\,
{\color{Orange} O^\nu_{\{\lambda\}}(\{\sigma\})}.
$$

Here, $\Lambda$ and $\nu$ indicate the allowed spin projections of the initial state, $\{\lambda\}$ are the allowed spin projections of the final state (e.g. $\{\lambda\}=\lambda_1,\lambda_3,\lambda_3$ for a three-body decay). The Euler angles $\alpha,\beta,\gamma$ are obtained by choosing a specific aligned center-of-momentum frame ("aligned CM"), see Fig.&nbsp;2 in Ref&nbsp;{cite}`JPAC:2019ufm`, which gives us an "aligned" transition amplitude $O^\nu_{\{\lambda\}}$ that only depends on dynamic variables $\{\sigma\}$ (in the case of a three-body decay, the three Mandelstam variables $\sigma_1,\sigma_2,\sigma_3$).

These aligned transition amplitudes are then combined into an observable **differential cross section** (intensity distribution), using a spin density matrix $\rho_{_{\Lambda,\Lambda'}}$ for the spin projections $\Lambda$ of the initial state,

$$
\mathrm{d}\sigma/\mathrm{d}\Phi_3 = N
\sum_{\Lambda,\Lambda'} \rho_{_{\Lambda,\Lambda'}}
\sum_{\nu,\nu'} {\color{RoyalBlue}
      D^{J^*}_{\Lambda,\nu}\left(\alpha,\beta,\gamma\right)
      D^{J}_{\Lambda',\nu'}\left(\alpha,\beta,\gamma\right)
   }
   \sum_{\{\lambda\}} {\color{Orange}
      O^\nu_{\{\lambda\}}\left(\{\sigma\}\right)
      O^{\nu'*}_{\{\lambda\}}\left(\{\sigma\}\right)
   }.
$$

Given the right alignment, the aligned transition amplitude can be written as

<!-- prettier-ignore-start -->

```{math}
---
label: aligned-amplitude
---
\begin{eqnarray*}
{\color{Orange}O^\nu_{\{\lambda\}}\left(\{\sigma\}\right)} &=&
   \sum_{(ij)k}
   \sum^{(ij)\to i,j}_s
   \sum_\tau
   \sum_{\{\lambda'\}}
   \; {\color{Orange}X_s\!\left(\sigma_k\right)}
\\ {\color{LightGray}\text{production:}} && \quad \times\;
   \eta_J\,
   d^J_{\nu,\tau-\lambda'_k}\!\left(\hat\theta_{k(1)}\right)\,
   H^{0\to(ij),k}_{\tau,\lambda_k'}\,(-1)^{j_k-\lambda_k'}
\\ {\color{LightGray}\text{decay:}} && \quad \times\;
   \eta_s\,
   d^s_{\tau,\lambda'_i-\lambda_j'}\!\left(\theta_{ij}\right)\,
   H^{(ij)\to i,j}_{\lambda'_i,\lambda'_j}\,(-1)^{j_j-\lambda_j'}
\\ {\color{LightGray}\text{rotations:}} && \quad \times\;
   d^{j_1}_{\lambda'_1,\lambda_1}\!\left(\zeta^1_{k(0)}\right)\,
   d^{j_2}_{\lambda'_2,\lambda_2}\!\left(\zeta^2_{k(0)}\right)\,
   d^{j_3}_{\lambda'_3,\lambda_3}\!\left(\zeta^3_{k(0)}\right).
\end{eqnarray*}
```

Notice the general structure:

- **Summations**: The outer sum is taken over the three decay chain combinations $(ij)k \in \left\{(23)1, (31)2, (12)3\right\}$. Next, we sum over the spin magnitudes&nbsp;$s$ of all resonances[^1], the corresponding allowed helicities&nbsp;$\tau$, and allowed spin projections&nbsp;$\{\lambda'\}$ of the final state.
- **Dynamics**: The function $X_s$ only depends on a single Mandelstam variable and carries all the dynamical information about the decay chain. Typically, these are your $K$-matrix or Breit-Wigner lineshape functions.
- **Isobars**: There is a Wigner&nbsp;$d$-function and a helicity coupling $H$ for each isobar in the three-body decay chain: the $0\to(ij),k$ production node and the $(ij)\to i,j$ decay node. The argument of these Wigner&nbsp;$d$-functions are the polar angles. The factors $\eta_J=\sqrt{2S+1}$ and $\eta_s=\sqrt{2s+1}$ are normalization factors. The phase $(-1)^{j-\lambda}$ is added to both helicity couplings to convert to the Jacob-Wick particle-2 convention.[^2] The convention treats the first and the second particle unequally, however, it enables the simple relation of the helicity couplings to the $LS$&nbsp;couplings explained below.
- **Wigner rotations**: The last three Wigner&nbsp;$d$-functions represent Wigner rotations that appear when rotating the boosted frames of the production and decay isobar amplitudes back to the space-fixed CM frame.

If $k=1$, we have $\hat\theta_{k(1)}=0$, so the Wigner&nbsp;$d$ function for the production isobar reduces to a Kronecker delta, $d^J_{\nu,\tau-\lambda'_k}\!\left(\hat\theta_{k(1)}\right) = \delta_{\nu,\tau-\lambda'_k}$.

[^1]: Alternatively, one can sum over all resonances themselves, so that one has a dynamic function&nbsp;$X_\mathcal{R}(\sigma_k)$ for each resonance&nbsp;$\mathcal{R}$ in subsystem&nbsp;$k$.

[^2]: See also {cite}`JPAC:2021rxu`, Section&nbsp;2.1.

Equation&nbsp;{eq}`aligned-amplitude` is written in terms of _helicity_ couplings, but can be rewritten in terms of _$LS$&nbsp;couplings_, using

```{math}
\begin{eqnarray*}
H^{0\to(ij),k}_{\tau,\lambda'_k} & = &
   \sum_{LS}
   H^{0\to(ij),k}_{LS}
   \sqrt{\frac{2L+1}{2J+1}}
   C^{S,\tau-\lambda'_k}_{s,\tau,j_k,-\lambda'_k}
   C^{J,\tau-\lambda'_k}_{L,0,S,\tau-\lambda'_k} \\
H^{(ij)\to i,j}_{\lambda'_i,\lambda'_j} & = &
   \sum_{l's'}
   H^{0\to(ij),k}_{l's'}
   \sqrt{\frac{2l'+1}{2s+1}}
   C^{s',\lambda'_i-\lambda'_j}_{j_i,\lambda'_i,j_j,-\lambda'_j}
   C^{s,\lambda'_i-\lambda'_j}_{l',0,s',\lambda'_i-\lambda'_j}\,.
\end{eqnarray*}
```

The dynamics function is dependent on the $LS$&nbsp;values and we write $X_s^{LS;l's'}$ instead of $X_s$.

## Examples

```{toctree}
---
maxdepth: 1
---
lc2pkpi
jpsi2ksp
xib2pkk
serialization
```

```{toctree}
---
maxdepth: 2
---
comparison
```

```{toctree}
---
hidden:
---
references
API <api/ampform_dpd>
```

```{toctree}
---
caption: Related projects
hidden:
---
QRules <https://qrules.readthedocs.io>
AmpForm <https://ampform.readthedocs.io>
TensorWaves <https://tensorwaves.readthedocs.io>
PWA Pages <https://pwa.readthedocs.io>
ComPWA project <https://compwa.github.io>
```
