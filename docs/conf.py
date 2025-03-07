from __future__ import annotations

from sphinx_api_relink.helpers import (
    get_branch_name,
    get_execution_mode,
    get_package_version,
    pin,
    pin_minor,
    set_intersphinx_version_remapping,
)

set_intersphinx_version_remapping({
    "ipython": {
        "8.12.2": "8.12.1",
        "8.12.3": "8.12.1",
    },
    "ipywidgets": {
        "8.0.3": "8.0.5",
        "8.0.4": "8.0.5",
        "8.0.6": "8.0.5",
        "8.1.1": "8.1.2",
    },
    "matplotlib": {"3.9.1.post1": "3.9.1"},
    "mpl-interactions": {"0.24.1": "0.24.0"},
})

BRANCH = get_branch_name()
ORGANIZATION = "ComPWA"
PACKAGE = "ampform_dpd"
REPO_NAME = "ampform-dpd"
REPO_TITLE = "Symbolic Dalitz-Plot Decomposition"

BINDER_LINK = (
    f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?urlpath=lab"
)
EXECUTE_NB = get_execution_mode() != "off"


add_module_names = False
api_github_repo = f"{ORGANIZATION}/{REPO_NAME}"
api_target_substitutions: dict[str, str | tuple[str, str]] = {
    "ampform_dpd.decay.StateIDTemplate": ("obj", "ampform_dpd.decay.StateID"),
    "ampform_dpd.io.serialization.dynamics.T": "typing.TypeVar",
    "DecayNode": ("obj", "ampform_dpd.decay.DecayNode"),
    "EdgeType": "typing.TypeVar",
    "FinalState": ("obj", "ampform_dpd.decay.FinalState"),
    "FinalStateID": ("obj", "ampform_dpd.decay.FinalStateID"),
    "FrozenTransition": "qrules.topology.FrozenTransition",
    "InitialStateID": ("obj", "ampform_dpd.decay.InitialStateID"),
    "Literal[-1, 1]": "typing.Literal",
    "Literal[(-1, 1)]": "typing.Literal",
    "Model": "ampform.sympy.cached.Model",
    "Node": ("obj", "ampform_dpd.io.serialization.format.Node"),
    "NodeType": "typing.TypeVar",
    "ParameterValue": ("obj", "tensorwaves.interface.ParameterValue"),
    "ParametrizedBackendFunction": "tensorwaves.function.ParametrizedBackendFunction",
    "PoolSum": "ampform.sympy.PoolSum",
    "PositionalArgumentFunction": "tensorwaves.function.PositionalArgumentFunction",
    "qrules.topology.EdgeType": "typing.TypeVar",
    "qrules.topology.NodeType": "typing.TypeVar",
    "sp.acos": "sympy.functions.elementary.trigonometric.acos",
    "sp.Basic": "sympy.core.basic.Basic",
    "sp.Expr": "sympy.core.expr.Expr",
    "sp.Indexed": "sympy.tensor.indexed.Indexed",
    "sp.Rational": "sympy.core.numbers.Rational",
    "sp.Symbol": "sympy.core.symbol.Symbol",
    "StateID": ("obj", "ampform_dpd.decay.StateID"),
    "StateIDTemplate": ("obj", "ampform_dpd.decay.StateID"),
    "Topology": ("obj", "ampform_dpd.io.serialization.format.Topology"),
    "typing_extensions.Required": ("obj", "typing.Required"),
}
api_target_types: dict[str, str] = {}
author = "Common Partial Wave Analysis"
autodoc_default_options = {
    "exclude-members": ", ".join([  # noqa: FLY002
        "default_assumptions",
        "doit",
        "evaluate",
        "is_commutative",
        "is_extended_real",
    ]),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt_et_al"
codeautolink_concat_default = True
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest
copyright = f"2022, {ORGANIZATION}"  # noqa: A001
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_book_theme",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_pybtex_etal_style",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
]
generate_apidoc_package_path = f"../src/{PACKAGE}"
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%-d %B %Y"
html_logo = (
    "https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg"
)
html_show_copyright = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "_static/favicon.ico",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{PACKAGE}",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Launch on Binder",
            "url": f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?urlpath=lab",
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/{ORGANIZATION}/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
    },
    "logo": {"text": REPO_TITLE},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = REPO_TITLE
intersphinx_mapping = {
    "IPython": (f"https://ipython.readthedocs.io/en/{pin('IPython')}", None),
    "ampform": (f"https://ampform.readthedocs.io/{pin('ampform')}", None),
    "attrs": (f"https://www.attrs.org/en/{pin('attrs')}", None),
    "compwa": ("https://compwa.github.io", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "ipywidgets": (f"https://ipywidgets.readthedocs.io/en/{pin('ipywidgets')}", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "matplotlib": (f"https://matplotlib.org/{pin('matplotlib')}", None),
    "numpy": (f"https://numpy.org/doc/{pin_minor('numpy')}", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (f"https://qrules.readthedocs.io/{pin('qrules')}", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": (f"https://tensorwaves.readthedocs.io/{pin('tensorwaves')}", None),
}
linkcheck_anchors = False
linkcheck_ignore = [
    "https://doi.org/10.1103",
    "https://journals.aps.org/prd",
]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_image",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3
myst_render_markdown_format = "myst"
myst_update_mathjax = False
nb_execution_allow_errors = False
nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "show"
nb_render_markdown_format = "myst"
nitpick_ignore = [
    ("py:class", "ampform.sympy.cached.Model"),
]
nitpicky = True
primary_domain = "py"
project = REPO_TITLE
pygments_style = "sphinx"
release = get_package_version(PACKAGE)
version = get_package_version(PACKAGE)
