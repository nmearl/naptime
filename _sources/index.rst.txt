NAPTIME
=======

**Neural Astrophysical Photometric Transient Identification and Modeling Engine**

NAPTIME is a ConvGNP-style neural-process framework for photometric transient
classification under sparse and partial observational context. It models irregular
multiband light curves directly, combines probabilistic light-curve reconstruction
with classification, and can incorporate host-galaxy context and photometric-redshift
information when available.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Quickstart
----------

Install with uv::

    uv sync

Run tests::

    uv run pytest

Build docs::

    uv run --group docs sphinx-build docs docs/_build/html

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
