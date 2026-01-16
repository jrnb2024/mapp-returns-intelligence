"""
Action definitions for the Returns Intelligence Agent.

Defines the Action dataclass and associated enums for representing
decisions made by the DecisionEngine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any
import uuid


class ActionType(Enum):
    """Types of actions the agent can recommend."""
    SUPPRESS = "SUPPRESS"  # Remove from recommendations (Dressipi not_features_ids)
    WARN = "WARN"          # Show sizing/fit warnings (Mapp Engage messaging)
    RESEGMENT = "RESEGMENT"  # Move customer between segments (Mapp Intelligence CI)


class ActionStatus(Enum):
    """Lifecycle status of an action."""
    PENDING = "PENDING"      # Decision made, awaiting approval
    APPROVED = "APPROVED"    # Human approved, ready to execute
    REJECTED = "REJECTED"    # Human rejected, will not execute
    EXECUTED = "EXECUTED"    # Successfully executed via API


@dataclass
class Action:
    """
    Represents a single decision/action recommended by the DecisionEngine.

    Attributes:
        id: UUID string for audit trail
        timestamp: When the decision was made
        action_type: SUPPRESS, WARN, or RESEGMENT
        target: SKU for SUPPRESS, product_id for WARN, customer_id for RESEGMENT
        confidence: Float 0.0-1.0 indicating decision confidence
        reason: Human-readable explanation of why this decision was made
        supporting_data: Dict with the metrics that drove the decision
        estimated_impact: Dict with return_reduction, cvr_impact, net_margin_impact
        mapp_api_call: Dict showing what API call would be made
        status: Current lifecycle status
    """
    action_type: ActionType
    target: str
    confidence: float
    reason: str
    supporting_data: Dict[str, Any]
    estimated_impact: Dict[str, float]
    mapp_api_call: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    status: ActionStatus = field(default=ActionStatus.PENDING)

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def approve(self) -> None:
        """Mark action as approved."""
        if self.status != ActionStatus.PENDING:
            raise ValueError(f"Can only approve PENDING actions, current status: {self.status}")
        self.status = ActionStatus.APPROVED

    def reject(self) -> None:
        """Mark action as rejected."""
        if self.status != ActionStatus.PENDING:
            raise ValueError(f"Can only reject PENDING actions, current status: {self.status}")
        self.status = ActionStatus.REJECTED

    def execute(self) -> None:
        """Mark action as executed."""
        if self.status != ActionStatus.APPROVED:
            raise ValueError(f"Can only execute APPROVED actions, current status: {self.status}")
        self.status = ActionStatus.EXECUTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'action_type': self.action_type.value,
            'target': self.target,
            'confidence': self.confidence,
            'reason': self.reason,
            'supporting_data': self.supporting_data,
            'estimated_impact': self.estimated_impact,
            'mapp_api_call': self.mapp_api_call,
            'status': self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create Action from dictionary."""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            action_type=ActionType(data['action_type']),
            target=data['target'],
            confidence=data['confidence'],
            reason=data['reason'],
            supporting_data=data['supporting_data'],
            estimated_impact=data['estimated_impact'],
            mapp_api_call=data['mapp_api_call'],
            status=ActionStatus(data['status']),
        )
