"""
Action Simulator for the Returns Intelligence Agent.

Generates detailed previews of what Mapp/Dressipi API calls would be made
when decisions are executed, including rollback procedures.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

from src.actions import Action, ActionType


class ActionSimulator:
    """
    Simulates API calls for decision execution.

    Provides detailed previews including:
    - Full API endpoint URLs
    - Request/response bodies
    - Estimated user impact
    - Rollback procedures
    """

    # API base URLs (illustrative - would be configured per environment)
    DRESSIPI_BASE = "https://api.dressipi.com/v1"
    MAPP_ENGAGE_BASE = "https://engage.mapp.com/api/v2"
    MAPP_INTELLIGENCE_BASE = "https://intelligence.mapp.com/api/v1"

    # Estimated daily traffic (would come from analytics in production)
    DAILY_RECOMMENDATION_REQUESTS = 2400
    AVG_DAILY_PDP_VIEWS_PER_PRODUCT = 150

    def __init__(self):
        """Initialize with Mapp API endpoint configurations."""
        self.execution_timestamp = datetime.now()

    def simulate(self, decision: Action) -> Dict[str, Any]:
        """
        Generate a detailed simulation of what would happen if this
        decision were executed.

        Args:
            decision: The Action to simulate

        Returns:
            Dictionary with simulation details including API endpoint,
            request body, expected response, affected users, and rollback procedure
        """
        if decision.action_type == ActionType.SUPPRESS:
            return self.simulate_suppress(decision)
        elif decision.action_type == ActionType.WARN:
            return self.simulate_warn(decision)
        elif decision.action_type == ActionType.RESEGMENT:
            return self.simulate_resegment(decision)
        else:
            raise ValueError(f"Unknown action type: {decision.action_type}")

    def simulate_suppress(self, decision: Action) -> Dict[str, Any]:
        """
        Simulate a SUPPRESS action via Dressipi API.

        Shows the facetted recommendations call with not_features_ids parameter.
        When a product is suppressed, it will be excluded from recommendation
        results across all channels.

        Args:
            decision: The SUPPRESS Action to simulate

        Returns:
            Simulation details for Dressipi API call
        """
        product_id = decision.target
        supporting_data = decision.supporting_data

        # Calculate estimated affected users based on product volume
        volume = supporting_data.get('volume', 100)
        volume_pct = volume / 50000  # Assuming 50k transactions in dataset
        affected_requests = int(self.DAILY_RECOMMENDATION_REQUESTS * volume_pct * 10)

        return {
            'action_type': 'SUPPRESS',
            'target': product_id,
            'api_endpoint': f"{self.DRESSIPI_BASE}/facetted/search",
            'http_method': 'POST',
            'request_body': {
                'user_id': '{{user_id}}',
                'session_id': '{{session_id}}',
                'facets': [
                    {'type': 'category', 'value': supporting_data.get('category', 'All')},
                ],
                'not_features_ids': [product_id],
                'limit': 24,
                'include_metadata': True,
            },
            'configuration_update': {
                'endpoint': f"{self.DRESSIPI_BASE}/config/exclusions",
                'method': 'PUT',
                'body': {
                    'exclusion_type': 'product',
                    'product_ids': [product_id],
                    'reason': 'high_return_rate',
                    'source': 'returns_intelligence_agent',
                    'decision_id': decision.id,
                    'expiry_date': (datetime.now() + timedelta(days=30)).isoformat(),
                },
            },
            'expected_response': {
                'status': 'success',
                'recommendations': '[...24 items excluding suppressed product...]',
                'excluded_count': 1,
                'exclusion_applied': True,
            },
            'affected_users': f"~{affected_requests:,} daily recommendation requests",
            'affected_products': 1,
            'duration': '30 days (configurable)',
            'rollback_procedure': {
                'description': 'Remove product ID from exclusions list',
                'endpoint': f"{self.DRESSIPI_BASE}/config/exclusions/{product_id}",
                'method': 'DELETE',
                'automatic_expiry': True,
            },
            'side_effects': [
                'Product will not appear in "You might also like" recommendations',
                'Product will not appear in email recommendation digests',
                'Product will still be searchable and accessible via direct link',
                'PDP cross-sell recommendations will exclude this product',
            ],
            'metrics_to_monitor': [
                'recommendation_click_through_rate',
                'category_conversion_rate',
                'return_rate_for_category',
            ],
        }

    def simulate_warn(self, decision: Action) -> Dict[str, Any]:
        """
        Simulate a WARN action via Mapp Engage.

        Shows the message template trigger that would display sizing warning
        on the product detail page.

        Args:
            decision: The WARN Action to simulate

        Returns:
            Simulation details for Mapp Engage API call
        """
        product_id = decision.target
        supporting_data = decision.supporting_data

        # Determine warning message based on sizing issue
        skew_direction = supporting_data.get('skew_direction', 'unknown')
        if skew_direction == 'small':
            template = 'sizing_warning_runs_small'
            headline = 'Sizing Note'
            body = 'This item runs small. Consider ordering one size up for your usual fit.'
            size_recommendation = '+1 size'
        elif skew_direction == 'large':
            template = 'sizing_warning_runs_large'
            headline = 'Sizing Note'
            body = 'This item runs large. Consider ordering one size down for your usual fit.'
            size_recommendation = '-1 size'
        else:
            template = 'sizing_warning_inconsistent'
            headline = 'Fit Guide'
            body = 'Customers report varied fit for this item. Check the size guide for measurements.'
            size_recommendation = 'Check size guide'

        # Estimate affected users
        volume = supporting_data.get('volume', 50)
        daily_views = max(10, int(volume / 30 * 3))  # Rough estimate

        trigger_id = f"trg_{uuid.uuid4().hex[:12]}"

        return {
            'action_type': 'WARN',
            'target': product_id,
            'api_endpoint': f"{self.MAPP_ENGAGE_BASE}/triggers",
            'http_method': 'POST',
            'request_body': {
                'trigger_type': 'product_page_warning',
                'trigger_name': f'sizing_warning_{product_id}',
                'product_id': product_id,
                'message_template': template,
                'content': {
                    'headline': headline,
                    'body': body,
                    'icon': 'ruler',
                    'style': 'info_banner',
                    'size_recommendation': size_recommendation,
                },
                'display_config': {
                    'location': 'pdp_size_selector',
                    'position': 'above',
                    'show_on_mobile': True,
                    'show_on_desktop': True,
                },
                'targeting': {
                    'all_users': True,
                    'exclude_segments': [],  # Could exclude repeat purchasers
                },
                'priority': 'high' if supporting_data.get('combined_sizing_issue', 0) > 0.5 else 'medium',
                'source': 'returns_intelligence_agent',
                'decision_id': decision.id,
            },
            'expected_response': {
                'trigger_id': trigger_id,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'estimated_impressions_per_day': daily_views,
            },
            'affected_users': f"~{daily_views} daily PDP views for this product",
            'affected_products': 1,
            'duration': 'Indefinite (until deactivated)',
            'rollback_procedure': {
                'description': 'Deactivate the trigger',
                'endpoint': f"{self.MAPP_ENGAGE_BASE}/triggers/{trigger_id}",
                'method': 'DELETE',
                'alternative': f"PATCH {self.MAPP_ENGAGE_BASE}/triggers/{trigger_id} with status=inactive",
            },
            'side_effects': [
                f'Sizing banner will appear on PDP for {product_id}',
                'May reduce conversion rate by 3-8% (customers reconsider)',
                'Expected to reduce returns by 10-20% for this product',
                'Warning will appear in all locales',
            ],
            'metrics_to_monitor': [
                'pdp_to_cart_conversion_rate',
                'product_return_rate',
                'size_exchange_requests',
                'customer_satisfaction_scores',
            ],
            'a_b_test_recommendation': {
                'description': 'Consider A/B testing warning message',
                'control': 'No warning',
                'variant': 'Sizing warning shown',
                'success_metric': 'return_rate',
                'minimum_sample': 500,
            },
        }

    def simulate_resegment(self, decision: Action) -> Dict[str, Any]:
        """
        Simulate a RESEGMENT action via Mapp Intelligence.

        Shows the segment membership update that moves a customer
        to a different return-risk segment.

        Args:
            decision: The RESEGMENT Action to simulate

        Returns:
            Simulation details for Mapp Intelligence API call
        """
        customer_id = decision.target
        supporting_data = decision.supporting_data

        current_segment = supporting_data.get('current_segment', 'unknown')
        new_segment = supporting_data.get('suggested_segment', 'high_returner')
        recent_returns = supporting_data.get('recent_returns', 0)
        period_days = supporting_data.get('period_days', 90)

        # Define segment treatment differences
        segment_treatments = {
            'serial_returner': {
                'recommendation_frequency': 'reduced',
                'email_frequency': 'weekly_digest_only',
                'promotion_eligibility': 'limited',
                'return_policy': 'standard',
                'personalization_strategy': 'high_confidence_only',
            },
            'high_returner': {
                'recommendation_frequency': 'standard',
                'email_frequency': 'standard',
                'promotion_eligibility': 'standard',
                'return_policy': 'standard',
                'personalization_strategy': 'conservative',
            },
        }

        treatment = segment_treatments.get(new_segment, {})

        return {
            'action_type': 'RESEGMENT',
            'target': customer_id,
            'api_endpoint': f"{self.MAPP_INTELLIGENCE_BASE}/segments/members",
            'http_method': 'PUT',
            'request_body': {
                'customer_id': customer_id,
                'operations': [
                    {
                        'action': 'remove',
                        'segment_name': current_segment,
                    },
                    {
                        'action': 'add',
                        'segment_name': new_segment,
                    },
                ],
                'reason': f'{recent_returns} returns in {period_days} days exceeds threshold',
                'source': 'returns_intelligence_agent',
                'decision_id': decision.id,
                'effective_date': datetime.now().isoformat(),
                'ttl_days': 90,  # Re-evaluate after 90 days
            },
            'expected_response': {
                'updated': True,
                'customer_id': customer_id,
                'previous_segment': current_segment,
                'new_segment': new_segment,
                'effective_date': datetime.now().isoformat(),
                'expiry_date': (datetime.now() + timedelta(days=90)).isoformat(),
            },
            'affected_users': '1 customer',
            'treatment_changes': treatment,
            'duration': '90 days (then re-evaluated)',
            'rollback_procedure': {
                'description': 'Reverse segment assignments',
                'endpoint': f"{self.MAPP_INTELLIGENCE_BASE}/segments/members",
                'method': 'PUT',
                'body': {
                    'customer_id': customer_id,
                    'operations': [
                        {'action': 'remove', 'segment_name': new_segment},
                        {'action': 'add', 'segment_name': current_segment},
                    ],
                    'reason': 'Rollback from returns intelligence agent',
                },
            },
            'side_effects': [
                f'Customer will receive {treatment.get("email_frequency", "modified")} email communications',
                f'Recommendation strategy changes to: {treatment.get("personalization_strategy", "modified")}',
                'Customer profile will show new segment in CRM',
                'Segment membership will auto-expire after 90 days if not renewed',
            ],
            'metrics_to_monitor': [
                'customer_return_rate_post_change',
                'customer_order_frequency',
                'customer_lifetime_value',
                'customer_churn_indicator',
            ],
            'gdpr_considerations': {
                'data_basis': 'Legitimate interest in fraud prevention',
                'customer_visible': False,  # Internal segmentation
                'right_to_explanation': True,
                'explanation_template': 'Your account has been flagged for review due to high return activity.',
            },
        }

    def generate_batch_simulation(self, decisions: List[Action]) -> Dict[str, Any]:
        """
        Generate a summary simulation for multiple decisions.

        Args:
            decisions: List of Actions to simulate

        Returns:
            Aggregate stats and sample of individual simulations
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        # Aggregate stats
        stats = {
            'batch_id': batch_id,
            'total_decisions': len(decisions),
            'generated_at': datetime.now().isoformat(),
            'by_type': {},
            'total_affected_products': 0,
            'total_affected_customers': 0,
            'api_calls_required': 0,
        }

        # Count by type
        for decision in decisions:
            action_type = decision.action_type.value
            if action_type not in stats['by_type']:
                stats['by_type'][action_type] = {
                    'count': 0,
                    'targets': [],
                }
            stats['by_type'][action_type]['count'] += 1
            stats['by_type'][action_type]['targets'].append(decision.target)

        # Calculate totals
        suppress_count = stats['by_type'].get('SUPPRESS', {}).get('count', 0)
        warn_count = stats['by_type'].get('WARN', {}).get('count', 0)
        resegment_count = stats['by_type'].get('RESEGMENT', {}).get('count', 0)

        stats['total_affected_products'] = suppress_count + warn_count
        stats['total_affected_customers'] = resegment_count
        stats['api_calls_required'] = len(decisions)  # One call per decision

        # Estimated execution time
        stats['estimated_execution_time_seconds'] = len(decisions) * 0.5  # 500ms per API call

        # Generate sample simulations (top 3 of each type)
        sample_simulations = []
        for action_type in [ActionType.SUPPRESS, ActionType.WARN, ActionType.RESEGMENT]:
            type_decisions = [d for d in decisions if d.action_type == action_type]
            for decision in type_decisions[:2]:  # Top 2 of each type
                sim = self.simulate(decision)
                sample_simulations.append({
                    'decision_id': decision.id,
                    'action_type': action_type.value,
                    'target': decision.target,
                    'api_endpoint': sim['api_endpoint'],
                    'http_method': sim['http_method'],
                    'affected_users': sim['affected_users'],
                })

        stats['sample_simulations'] = sample_simulations

        # Execution plan
        stats['execution_plan'] = {
            'recommended_order': [
                '1. Execute SUPPRESS actions (immediate impact on recommendations)',
                '2. Execute WARN actions (requires content deployment)',
                '3. Execute RESEGMENT actions (batch update to customer profiles)',
            ],
            'parallel_execution_possible': True,
            'rate_limit_considerations': 'Mapp API: 100 requests/minute',
            'suggested_batch_size': 50,
        }

        # Rollback summary
        stats['batch_rollback'] = {
            'endpoint': f"{self.MAPP_INTELLIGENCE_BASE}/batches/{batch_id}/rollback",
            'method': 'POST',
            'description': 'Rolls back all actions in this batch',
            'available_for': '30 days after execution',
        }

        return stats

    def preview_execution_script(self, decisions: List[Action]) -> str:
        """
        Generate a preview of the execution script (for review).

        Args:
            decisions: List of Actions

        Returns:
            String containing pseudo-code execution script
        """
        lines = [
            "# Returns Intelligence Agent - Execution Preview",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Total decisions: {len(decisions)}",
            "",
            "from mapp_sdk import MappClient, DressIPiClient",
            "",
            "mapp = MappClient(api_key=MAPP_API_KEY)",
            "dressipi = DressipiClient(api_key=DRESSIPI_API_KEY)",
            "",
        ]

        for i, decision in enumerate(decisions[:10]):  # Show first 10
            lines.append(f"# Decision {i+1}: {decision.action_type.value} - {decision.target}")
            lines.append(f"# Confidence: {decision.confidence:.0%}")
            lines.append(f"# Reason: {decision.reason[:60]}...")

            if decision.action_type == ActionType.SUPPRESS:
                lines.append(f"dressipi.exclude_product('{decision.target}', duration_days=30)")
            elif decision.action_type == ActionType.WARN:
                lines.append(f"mapp.create_trigger('sizing_warning', product_id='{decision.target}')")
            elif decision.action_type == ActionType.RESEGMENT:
                lines.append(f"mapp.update_segment('{decision.target}', segment='{decision.supporting_data.get('suggested_segment')}')")

            lines.append("")

        if len(decisions) > 10:
            lines.append(f"# ... and {len(decisions) - 10} more decisions")

        return "\n".join(lines)
