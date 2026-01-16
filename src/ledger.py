"""
Decision Ledger for the Returns Intelligence Agent.

Provides persistent storage for decisions with full audit trail,
querying capabilities, and impact tracking.
"""

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.actions import Action, ActionType, ActionStatus


class DecisionLedger:
    """
    Persists decisions to JSON file with querying and audit capabilities.

    The ledger tracks:
    - All decisions with their full context
    - Status transitions (PENDING -> APPROVED -> EXECUTED)
    - Timestamps for recording, approval, and execution
    - Cumulative impact metrics
    """

    def __init__(self, ledger_path: str = "data/ledger.json"):
        """
        Initialize ledger, loading existing data if present.

        Args:
            ledger_path: Path to the JSON file for persistence
        """
        self.ledger_path = ledger_path
        self._decisions: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0',
        }
        self._load()

    def _load(self) -> None:
        """Load existing ledger data from file."""
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r') as f:
                    data = json.load(f)
                    self._decisions = data.get('decisions', {})
                    self._metadata = data.get('metadata', self._metadata)
            except (json.JSONDecodeError, IOError):
                # Start fresh if file is corrupted
                self._decisions = {}

    def _save(self) -> None:
        """Persist ledger data to file."""
        self._metadata['last_updated'] = datetime.now().isoformat()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)

        data = {
            'metadata': self._metadata,
            'decisions': self._decisions,
        }

        with open(self.ledger_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def record_decision(self, decision: Action) -> None:
        """
        Add a decision to the ledger with timestamp.

        Args:
            decision: The Action to record
        """
        record = decision.to_dict()
        record['recorded_at'] = datetime.now().isoformat()
        record['status_history'] = [
            {
                'status': decision.status.value,
                'timestamp': datetime.now().isoformat(),
                'note': 'Initial recording',
            }
        ]

        self._decisions[decision.id] = record
        self._save()

    def record_decisions(self, decisions: List[Action]) -> None:
        """
        Bulk add decisions to the ledger.

        Args:
            decisions: List of Actions to record
        """
        for decision in decisions:
            record = decision.to_dict()
            record['recorded_at'] = datetime.now().isoformat()
            record['status_history'] = [
                {
                    'status': decision.status.value,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Initial recording',
                }
            ]
            self._decisions[decision.id] = record

        self._save()

    def get_decision(self, decision_id: str) -> Optional[Action]:
        """
        Retrieve a specific decision by ID.

        Args:
            decision_id: The UUID of the decision

        Returns:
            The Action if found, None otherwise
        """
        record = self._decisions.get(decision_id)
        if record is None:
            return None

        return Action.from_dict(record)

    def get_decisions(
        self,
        action_type: Optional[ActionType] = None,
        status: Optional[ActionStatus] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Action]:
        """
        Query decisions with filters.

        Args:
            action_type: Filter by action type (SUPPRESS, WARN, RESEGMENT)
            status: Filter by status (PENDING, APPROVED, etc.)
            since: Filter by timestamp >= since
            until: Filter by timestamp <= until
            min_confidence: Filter by confidence >= min_confidence
            limit: Maximum number of results to return

        Returns:
            List of matching Actions, sorted by confidence (descending)
        """
        results = []

        for record in self._decisions.values():
            # Apply filters
            if action_type is not None and record['action_type'] != action_type.value:
                continue

            if status is not None and record['status'] != status.value:
                continue

            if min_confidence is not None and record['confidence'] < min_confidence:
                continue

            record_time = datetime.fromisoformat(record['timestamp'])

            if since is not None and record_time < since:
                continue

            if until is not None and record_time > until:
                continue

            results.append(Action.from_dict(record))

        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    def update_status(
        self,
        decision_id: str,
        new_status: ActionStatus,
        note: Optional[str] = None,
    ) -> bool:
        """
        Update a decision's status (e.g., PENDING -> APPROVED).

        Args:
            decision_id: The UUID of the decision
            new_status: The new status to set
            note: Optional note explaining the status change

        Returns:
            True if update succeeded, False if decision not found
        """
        if decision_id not in self._decisions:
            return False

        record = self._decisions[decision_id]
        old_status = record['status']

        # Update status
        record['status'] = new_status.value

        # Add to status history
        if 'status_history' not in record:
            record['status_history'] = []

        record['status_history'].append({
            'status': new_status.value,
            'previous_status': old_status,
            'timestamp': datetime.now().isoformat(),
            'note': note or f'Status changed from {old_status} to {new_status.value}',
        })

        # Track specific timestamps
        if new_status == ActionStatus.APPROVED:
            record['approved_at'] = datetime.now().isoformat()
        elif new_status == ActionStatus.EXECUTED:
            record['executed_at'] = datetime.now().isoformat()
        elif new_status == ActionStatus.REJECTED:
            record['rejected_at'] = datetime.now().isoformat()

        self._save()
        return True

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Return aggregate stats: total decisions, by type, by status, etc.

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_decisions': len(self._decisions),
            'by_type': {},
            'by_status': {},
            'by_confidence_tier': {
                'high': 0,    # >= 0.8
                'medium': 0,  # >= 0.6
                'low': 0,     # < 0.6
            },
            'avg_confidence': 0.0,
            'ledger_created': self._metadata.get('created_at'),
            'last_updated': self._metadata.get('last_updated'),
        }

        if not self._decisions:
            return stats

        total_confidence = 0.0

        for record in self._decisions.values():
            # By type
            action_type = record['action_type']
            stats['by_type'][action_type] = stats['by_type'].get(action_type, 0) + 1

            # By status
            status = record['status']
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # By confidence tier
            confidence = record['confidence']
            total_confidence += confidence

            if confidence >= 0.8:
                stats['by_confidence_tier']['high'] += 1
            elif confidence >= 0.6:
                stats['by_confidence_tier']['medium'] += 1
            else:
                stats['by_confidence_tier']['low'] += 1

        stats['avg_confidence'] = total_confidence / len(self._decisions)

        return stats

    def get_cumulative_impact(self) -> Dict[str, Any]:
        """
        Sum up estimated_impact across all APPROVED/EXECUTED decisions.

        Returns:
            Dictionary with cumulative impact metrics
        """
        impact = {
            'total_decisions_counted': 0,
            'total_return_reduction': 0.0,
            'total_cvr_impact': 0.0,
            'total_net_margin_impact': 0.0,
            'by_type': {},
        }

        actionable_statuses = {ActionStatus.APPROVED.value, ActionStatus.EXECUTED.value}

        for record in self._decisions.values():
            if record['status'] not in actionable_statuses:
                continue

            impact['total_decisions_counted'] += 1

            estimated = record.get('estimated_impact', {})
            impact['total_return_reduction'] += estimated.get('return_reduction', 0)
            impact['total_cvr_impact'] += estimated.get('cvr_impact', 0)
            impact['total_net_margin_impact'] += estimated.get('net_margin_impact', 0)

            # Track by type
            action_type = record['action_type']
            if action_type not in impact['by_type']:
                impact['by_type'][action_type] = {
                    'count': 0,
                    'net_margin_impact': 0.0,
                }
            impact['by_type'][action_type]['count'] += 1
            impact['by_type'][action_type]['net_margin_impact'] += estimated.get('net_margin_impact', 0)

        return impact

    def approve_decisions(
        self,
        decision_ids: Optional[List[str]] = None,
        action_type: Optional[ActionType] = None,
        min_confidence: Optional[float] = None,
    ) -> int:
        """
        Bulk approve decisions matching criteria.

        Args:
            decision_ids: Specific IDs to approve (if None, uses filters)
            action_type: Filter by action type
            min_confidence: Only approve if confidence >= threshold

        Returns:
            Number of decisions approved
        """
        count = 0

        for decision_id, record in self._decisions.items():
            # Skip if not pending
            if record['status'] != ActionStatus.PENDING.value:
                continue

            # Apply filters
            if decision_ids is not None and decision_id not in decision_ids:
                continue

            if action_type is not None and record['action_type'] != action_type.value:
                continue

            if min_confidence is not None and record['confidence'] < min_confidence:
                continue

            # Approve
            self.update_status(decision_id, ActionStatus.APPROVED, 'Bulk approval')
            count += 1

        return count

    def clear(self) -> None:
        """Clear all decisions (for testing)."""
        self._decisions = {}
        self._metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0',
        }
        self._save()

    def export_for_review(self, filepath: str) -> None:
        """
        Export decisions to a human-readable format for review.

        Args:
            filepath: Path to save the export
        """
        export_data = {
            'export_time': datetime.now().isoformat(),
            'summary': self.get_summary_stats(),
            'decisions': [],
        }

        for record in self._decisions.values():
            export_data['decisions'].append({
                'id': record['id'],
                'type': record['action_type'],
                'target': record['target'],
                'confidence': f"{record['confidence']:.0%}",
                'status': record['status'],
                'reason': record['reason'],
                'estimated_margin_impact': f"£{record['estimated_impact'].get('net_margin_impact', 0):,.2f}",
            })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
