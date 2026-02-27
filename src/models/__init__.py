"""
StructRouter Model Components

对应图中的主要模块:
- Learnable Structure Encoder (TCN-based)
- Multi-Pattern Weight Estimator
- Cognitive Router (Core Innovation & Orchestration)
- Task Expert Pool
- Verification Agent
- Segment Fusion
"""

from .structure_encoder import StructureEncoder
from .weight_estimator import MultiPatternWeightEstimator
from .experts import (
    ExpertBase,
    PeriodicExpert,
    TrendExpert,
    NoiseExpert,
    AbruptExpert,
    ChangePointExpert,  # 别名
    GeneralExpert
)
from .expert_fusion import SoftWeightedExpertFusion
from .tool_adapter import ToolAdapterRFT
from .causal_communication import CausalCommunicationSCM, TemporalCausalSCM
from .verification import ClosedLoopVerification
from .verification_agent import VerificationAgent
from .segment_processor import (
    SegmentDetector,
    SegmentLevelWeightEstimator,
    PatternAbstraction,
    MultiSegmentRouteComposition,
    SegmentFusion
)
from .struct_router import StructRouter

__all__ = [
    # Encoder
    'StructureEncoder',
    'MultiPatternWeightEstimator',
    # Experts
    'ExpertBase',
    'PeriodicExpert',
    'TrendExpert',
    'NoiseExpert',
    'AbruptExpert',
    'ChangePointExpert',
    'GeneralExpert',
    'SoftWeightedExpertFusion',
    # Routing
    'ToolAdapterRFT',
    # Communication
    'CausalCommunicationSCM',
    'TemporalCausalSCM',
    # Verification
    'ClosedLoopVerification',
    'VerificationAgent',
    # Segment Processing
    'SegmentDetector',
    'SegmentLevelWeightEstimator',
    'PatternAbstraction',
    'MultiSegmentRouteComposition',
    'SegmentFusion',
    # Main Model
    'StructRouter'
]
