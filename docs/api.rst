API Reference
=============

.. contents:: Modules
   :local:
   :depth: 1

Neural Network Modules
----------------------

Core building blocks: time embedding, set convolution, backbone, latent encoder, and MLP.

.. automodule:: naptime.modules
   :members:
   :undoc-members: False
   :show-inheritance:

Model
-----

The ``ConvGNPBaseline`` model, its configuration, batch dataclass, and training utilities.

.. automodule:: naptime.baseline
   :members: ConvGNPBaselineConfig, ConvGNPBaseline, ConvGNPBaselineOutput,
             BaselineBatch, BaselineLossConfig, BaselineLosses,
             baseline_loss, fit_epoch, evaluate_epoch,
             save_baseline_checkpoint, load_baseline_checkpoint
   :undoc-members: False
   :show-inheritance:

ELAsTiCC Taxonomy
-----------------

Family-level taxonomy constants and release-directory mappings.

.. automodule:: naptime.elasticc
   :members:
   :undoc-members: False

Data Utilities
--------------

Light-curve loading, normalization, and EBV correction helpers.

.. automodule:: naptime.data
   :members:
   :undoc-members: False
