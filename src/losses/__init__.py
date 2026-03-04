"""
StructRouter Loss Functions
"""

from .joint_loss import (
    JointLoss,
    TaskLoss,
    StructureConsistencyLoss,
    SparsityLoss,
    DAGConstraintLoss,
    LoadBalanceLoss,
    PrototypeOrthogonalityLoss,
    ExpertBalanceVarianceLoss,
)

__all__ = [
    'JointLoss',
    'TaskLoss',
    'StructureConsistencyLoss',
    'SparsityLoss',
    'DAGConstraintLoss',
    'LoadBalanceLoss',
    'PrototypeOrthogonalityLoss',
    'ExpertBalanceVarianceLoss',
]
