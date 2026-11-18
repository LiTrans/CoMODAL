"""Learned-routing baselines for the bimodal UAV/ADR CE-CPDPTW setting.

Published-method baselines adapted to this problem setting. Each baseline:

- consumes the same batch dict as ``VRP_Actor.Model`` (keys: ``x``, ``capacity``,
  ``demand``, ``battery``, ``time_window``, ``n_depots``, ``edge_attr_d``,
  ``edge_attr_r``, ``edge_attr_uav``, ``edge_attr_adr``, ``edge_index``,
  ``mask_adjacency_uav``, ``mask_adjacency_adr``, ...);
- returns the same ``(actions, log_p, time)`` 3-tuple (or 4-tuple when
  ``return_entropy=True``);
- is a drop-in replacement for ``Model`` in the existing training harness.

Operational constraints (mandatory service, battery, time windows, pickup/
delivery pairing) are enforced by the shared ``CoopDecoder`` physics stack —
baselines differ only in their *encoder*. This isolates the architectural
contribution claimed for HetGAT: the heterogeneous cross-fleet GAT message
passing vs. plainer attention stacks.
"""

from .kool_am import KoolAMEncoder, KoolAMModel
from .hetero_attn_pdp import HAPDPEncoder, HAPDPModel
from .gann import GANNEncoder, GANNModel

__all__ = [
    'KoolAMEncoder', 'KoolAMModel',
    'HAPDPEncoder',  'HAPDPModel',
    'GANNEncoder',   'GANNModel',
]
