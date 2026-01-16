"""
Decision Engine for the Returns Intelligence Agent.

Converts analysis insights from ReturnsAnalyzer into actionable decisions
that can be executed via Mapp/Dressipi APIs.
"""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.analysis_engine import ReturnsAnalyzer
from src.actions import Action, ActionType, ActionStatus


class DecisionEngine:
    """
    Generates actionable decisions based on returns analysis.

    Decision types:
    - SUPPRESS: Remove high-return products from recommendations
    - WARN: Add sizing/fit warnings to product pages
    - RESEGMENT: Move customers between return-risk segments
    """

    def __init__(self, analyzer: ReturnsAnalyzer, config_path: Optional[str] = None):
        """
        Initialize DecisionEngine with a ReturnsAnalyzer instance.

        Args:
            analyzer: ReturnsAnalyzer instance with transaction data
            config_path: Optional path to thresholds.yaml config file
        """
        self.analyzer = analyzer
        self.thresholds = self._load_thresholds(config_path)

    def _load_thresholds(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load thresholds from YAML config file."""
        if config_path is None:
            # Default path relative to project root
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'thresholds.yaml'
            )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Fallback defaults
            return {
                'suppress': {
                    'min_return_rate': 0.45,
                    'min_volume': 30,
                    'excess_multiplier': 2.0,
                },
                'warn': {
                    'min_sizing_issue_rate': 0.40,
                    'min_return_rate': 0.35,
                },
                'resegment': {
                    'serial_returner_threshold': 5,
                    'high_returner_rate': 0.50,
                    'period_days': 90,
                },
                'general': {
                    'min_confidence_to_act': 0.70,
                },
            }

    def generate_decisions(self) -> List[Action]:
        """
        Generate all decisions based on current analysis.

        Returns:
            List of Action objects representing recommended decisions
        """
        decisions = []

        # Generate each type of decision
        decisions.extend(self._generate_suppress_decisions())
        decisions.extend(self._generate_warn_decisions())
        decisions.extend(self._generate_resegment_decisions())

        # Sort by confidence (highest first)
        decisions.sort(key=lambda x: x.confidence, reverse=True)

        return decisions

    def _generate_suppress_decisions(self) -> List[Action]:
        """
        Generate SUPPRESS decisions for high-volume, high-return products.

        Rule: SUPPRESS if:
        - return_rate > min_return_rate (default 0.45)
        - volume > min_volume (default 30)
        - return_rate > excess_multiplier * category_average (default 2x)
        """
        decisions = []
        thresholds = self.thresholds['suppress']

        # Get HVHR products from analyzer
        hvhr_products = self.analyzer.get_hvhr_products(
            min_qty=thresholds['min_volume'],
            top_n=100
        )

        if len(hvhr_products) == 0:
            return decisions

        # Get category baselines for context
        category_analysis = self.analyzer.get_category_analysis()

        for _, product in hvhr_products.iterrows():
            return_rate = product['return_rate']
            category_baseline = product['category_baseline']
            volume = product['qty_purchased']

            # Check thresholds
            if return_rate < thresholds['min_return_rate']:
                continue
            if volume < thresholds['min_volume']:
                continue
            if return_rate < (thresholds['excess_multiplier'] * category_baseline):
                continue

            # Calculate confidence based on sample size
            # Higher volume = higher confidence (log scale)
            confidence = min(0.95, 0.5 + 0.15 * np.log10(volume))

            # Calculate multiplier vs category
            multiplier = return_rate / category_baseline if category_baseline > 0 else 0

            # Build reason string
            reason = (
                f"{product['product_name']} has {return_rate:.0%} return rate, "
                f"{multiplier:.1f}x the category average of {category_baseline:.0%}"
            )

            # Supporting data
            supporting_data = {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'],
                'return_rate': float(return_rate),
                'category_baseline': float(category_baseline),
                'excess_return_rate': float(product['excess_return_rate']),
                'volume': int(volume),
                'returns_count': int(product['qty_returned']),
                'multiplier_vs_category': float(multiplier),
            }

            # Estimated impact
            # If we suppress, we expect to reduce returns by the excess amount
            excess_returns = volume * product['excess_return_rate']
            return_reduction = excess_returns / self.analyzer.transactions['qty_returned'].sum()

            estimated_impact = {
                'return_reduction': float(return_reduction),
                'cvr_impact': -0.02,  # Small negative CVR impact from removing product
                'net_margin_impact': float(excess_returns * self.analyzer.COST_PER_RETURN),
            }

            # Mapp API call structure (Dressipi not_features_ids)
            mapp_api_call = {
                'endpoint': 'dressipi/not_features_ids',
                'method': 'POST',
                'params': {
                    'product_id': product['product_id'],
                    'action': 'suppress',
                    'duration_days': 30,
                    'reason': 'high_return_rate',
                },
            }

            action = Action(
                action_type=ActionType.SUPPRESS,
                target=product['product_id'],
                confidence=confidence,
                reason=reason,
                supporting_data=supporting_data,
                estimated_impact=estimated_impact,
                mapp_api_call=mapp_api_call,
            )

            decisions.append(action)

        return decisions

    def _generate_warn_decisions(self) -> List[Action]:
        """
        Generate WARN decisions for products with sizing issues.

        Rule: WARN if:
        - Product has sizing_inconsistency (>40% of returns from size issues)
        - return_rate > min_return_rate (default 0.35)

        Sizing issues detected via:
        - High proportion of multi-size order returns
        - Extreme size distribution in returns vs purchases
        """
        decisions = []
        thresholds = self.thresholds['warn']

        # Analyze sizing patterns at product level
        transactions = self.analyzer.transactions

        # Get products with multi-size order patterns
        product_sizing = transactions.groupby(['product_id', 'product_name', 'category']).agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'is_multi_size_order': 'sum',  # Count of multi-size order items
        }).reset_index()

        product_sizing['return_rate'] = (
            product_sizing['qty_returned'] / product_sizing['qty_purchased']
        )

        # Calculate sizing issue rate (proxy: multi-size orders with returns)
        multi_size_returns = transactions[
            (transactions['is_multi_size_order'] == True) &
            (transactions['qty_returned'] > 0)
        ].groupby('product_id')['qty_returned'].sum()

        product_sizing['multi_size_returns'] = product_sizing['product_id'].map(
            multi_size_returns
        ).fillna(0)

        product_sizing['sizing_issue_rate'] = np.where(
            product_sizing['qty_returned'] > 0,
            product_sizing['multi_size_returns'] / product_sizing['qty_returned'],
            0
        )

        # Also check for size skew in returns
        # Products where returns are concentrated in specific sizes
        size_skew = self._calculate_size_skew_by_product(transactions)
        product_sizing = product_sizing.merge(
            size_skew[['product_id', 'size_skew_score', 'dominant_return_size', 'skew_direction']],
            on='product_id',
            how='left'
        )
        product_sizing['size_skew_score'] = product_sizing['size_skew_score'].fillna(0)

        # Combined sizing issue indicator
        product_sizing['combined_sizing_issue'] = (
            product_sizing['sizing_issue_rate'] * 0.5 +
            product_sizing['size_skew_score'] * 0.5
        )

        # Filter products meeting thresholds
        candidates = product_sizing[
            (product_sizing['return_rate'] >= thresholds['min_return_rate']) &
            (product_sizing['combined_sizing_issue'] >= thresholds['min_sizing_issue_rate']) &
            (product_sizing['qty_purchased'] >= 20)  # Minimum sample
        ]

        for _, product in candidates.iterrows():
            # Calculate confidence
            volume = product['qty_purchased']
            confidence = min(0.90, 0.4 + 0.20 * np.log10(max(volume, 1)))

            # Determine the type of sizing issue
            skew_direction = product.get('skew_direction', 'unknown')
            dominant_size = product.get('dominant_return_size', 'unknown')

            if skew_direction == 'small':
                sizing_message = f"runs small (customers frequently return for larger size)"
            elif skew_direction == 'large':
                sizing_message = f"runs large (customers frequently return for smaller size)"
            else:
                sizing_message = f"has inconsistent sizing"

            # Build reason
            reason = (
                f"{product['product_name']} {sizing_message}. "
                f"{product['return_rate']:.0%} return rate with "
                f"{product['combined_sizing_issue']:.0%} sizing-related issues."
            )

            supporting_data = {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'],
                'return_rate': float(product['return_rate']),
                'sizing_issue_rate': float(product['sizing_issue_rate']),
                'size_skew_score': float(product['size_skew_score']),
                'combined_sizing_issue': float(product['combined_sizing_issue']),
                'dominant_return_size': str(dominant_size),
                'skew_direction': str(skew_direction),
                'volume': int(volume),
            }

            # Estimated impact - warnings typically reduce returns by 5-15%
            current_returns = product['qty_returned']
            expected_reduction = current_returns * 0.10  # 10% reduction estimate

            estimated_impact = {
                'return_reduction': float(expected_reduction / self.analyzer.transactions['qty_returned'].sum()),
                'cvr_impact': -0.01,  # Small negative from friction
                'net_margin_impact': float(expected_reduction * self.analyzer.COST_PER_RETURN),
            }

            # Mapp API call structure (Mapp Engage messaging)
            warning_text = f"This item {sizing_message}. Check size guide for best fit."

            mapp_api_call = {
                'endpoint': 'mapp-engage/product-messaging',
                'method': 'POST',
                'params': {
                    'product_id': product['product_id'],
                    'message_type': 'sizing_warning',
                    'message_text': warning_text,
                    'display_location': 'pdp_size_selector',
                    'priority': 'high' if product['combined_sizing_issue'] > 0.5 else 'medium',
                },
            }

            action = Action(
                action_type=ActionType.WARN,
                target=product['product_id'],
                confidence=confidence,
                reason=reason,
                supporting_data=supporting_data,
                estimated_impact=estimated_impact,
                mapp_api_call=mapp_api_call,
            )

            decisions.append(action)

        return decisions

    def _calculate_size_skew_by_product(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate size skew for each product.

        Returns DataFrame with product_id, size_skew_score, dominant_return_size, skew_direction.
        """
        # Filter to products with returns
        returned = transactions[transactions['qty_returned'] > 0].copy()

        if len(returned) == 0:
            return pd.DataFrame(columns=['product_id', 'size_skew_score', 'dominant_return_size', 'skew_direction'])

        # Define size order for determining direction
        numeric_order = {'4': 0, '6': 1, '8': 2, '10': 3, '12': 4, '14': 5, '16': 6, '18': 7}
        alpha_order = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4}

        def get_size_order(size):
            if size in numeric_order:
                return numeric_order[size]
            if size in alpha_order:
                return alpha_order[size]
            return -1

        # Calculate return distribution by size for each product
        product_size_returns = returned.groupby(['product_id', 'size'])['qty_returned'].sum().reset_index()
        product_total_returns = returned.groupby('product_id')['qty_returned'].sum().reset_index()
        product_total_returns.columns = ['product_id', 'total_returns']

        product_size_returns = product_size_returns.merge(product_total_returns, on='product_id')
        product_size_returns['return_pct'] = (
            product_size_returns['qty_returned'] / product_size_returns['total_returns']
        )

        # Find dominant return size and skew
        results = []
        for product_id in product_size_returns['product_id'].unique():
            prod_data = product_size_returns[product_size_returns['product_id'] == product_id]

            if len(prod_data) < 2:
                continue

            # Find size with highest return percentage
            dominant = prod_data.loc[prod_data['return_pct'].idxmax()]
            dominant_size = dominant['size']
            dominant_pct = dominant['return_pct']

            # Calculate skew score (how concentrated returns are)
            # Higher concentration = higher sizing issue
            skew_score = dominant_pct - (1 / len(prod_data))  # vs uniform distribution
            skew_score = max(0, skew_score)

            # Determine direction
            size_order = get_size_order(dominant_size)
            if dominant_size in numeric_order:
                midpoint = 3.5  # Between 10 and 12
            elif dominant_size in alpha_order:
                midpoint = 2  # M
            else:
                midpoint = -1

            if size_order >= 0 and midpoint >= 0:
                if size_order < midpoint:
                    direction = 'small'  # Small sizes returned more = runs small
                elif size_order > midpoint:
                    direction = 'large'  # Large sizes returned more = runs large
                else:
                    direction = 'unknown'
            else:
                direction = 'unknown'

            results.append({
                'product_id': product_id,
                'size_skew_score': skew_score,
                'dominant_return_size': dominant_size,
                'skew_direction': direction,
            })

        return pd.DataFrame(results)

    def _generate_resegment_decisions(self) -> List[Action]:
        """
        Generate RESEGMENT decisions for customers needing segment changes.

        Rules:
        - If customer has 5+ returns in last 90 days -> serial_returner
        - If customer return_rate > 0.50 -> high_returner
        """
        decisions = []
        thresholds = self.thresholds['resegment']

        transactions = self.analyzer.transactions

        # Calculate customer metrics
        # First, get recent activity (last N days)
        if 'order_date' in transactions.columns:
            max_date = transactions['order_date'].max()
            cutoff_date = max_date - timedelta(days=thresholds['period_days'])
            recent = transactions[transactions['order_date'] >= cutoff_date]
        else:
            recent = transactions  # Use all data if no dates

        # Customer stats from recent period
        customer_recent = recent.groupby('customer_id').agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'customer_segment': 'first',
            'order_id': 'nunique',
        }).reset_index()
        customer_recent.columns = ['customer_id', 'recent_purchases', 'recent_returns',
                                   'current_segment', 'recent_orders']

        customer_recent['recent_return_rate'] = np.where(
            customer_recent['recent_purchases'] > 0,
            customer_recent['recent_returns'] / customer_recent['recent_purchases'],
            0
        )

        # Rule 1: Serial returner (5+ returns in period)
        serial_candidates = customer_recent[
            (customer_recent['recent_returns'] >= thresholds['serial_returner_threshold']) &
            (customer_recent['current_segment'] != 'serial_returner')
        ]

        for _, customer in serial_candidates.iterrows():
            # Confidence based on return count
            returns = customer['recent_returns']
            confidence = min(0.95, 0.6 + 0.05 * (returns - thresholds['serial_returner_threshold']))

            reason = (
                f"Customer {customer['customer_id']} has made {int(returns)} returns "
                f"in the last {thresholds['period_days']} days "
                f"(threshold: {thresholds['serial_returner_threshold']}). "
                f"Current segment: {customer['current_segment']}."
            )

            supporting_data = {
                'customer_id': customer['customer_id'],
                'current_segment': customer['current_segment'],
                'suggested_segment': 'serial_returner',
                'recent_returns': int(returns),
                'recent_purchases': int(customer['recent_purchases']),
                'recent_return_rate': float(customer['recent_return_rate']),
                'recent_orders': int(customer['recent_orders']),
                'period_days': thresholds['period_days'],
            }

            # Estimated impact - serial returners typically have 90%+ return rates
            # Moving them to segment allows personalized treatment
            estimated_impact = {
                'return_reduction': 0.001,  # Per-customer impact is small
                'cvr_impact': 0.0,
                'net_margin_impact': float(returns * self.analyzer.COST_PER_RETURN * 0.2),
            }

            mapp_api_call = {
                'endpoint': 'mapp-intelligence/ci-segments',
                'method': 'PUT',
                'params': {
                    'customer_id': customer['customer_id'],
                    'segment_action': 'add',
                    'segment_name': 'serial_returner',
                    'reason': 'automated_resegment',
                    'ttl_days': 90,
                },
            }

            action = Action(
                action_type=ActionType.RESEGMENT,
                target=customer['customer_id'],
                confidence=confidence,
                reason=reason,
                supporting_data=supporting_data,
                estimated_impact=estimated_impact,
                mapp_api_call=mapp_api_call,
            )

            decisions.append(action)

        # Rule 2: High returner (return_rate > 0.50)
        high_returner_candidates = customer_recent[
            (customer_recent['recent_return_rate'] >= thresholds['high_returner_rate']) &
            (customer_recent['recent_purchases'] >= 5) &  # Minimum sample
            (~customer_recent['current_segment'].isin(['serial_returner', 'high_returner']))
        ]

        for _, customer in high_returner_candidates.iterrows():
            # Confidence based on sample size and return rate
            purchases = customer['recent_purchases']
            return_rate = customer['recent_return_rate']
            confidence = min(0.90, 0.5 + 0.10 * np.log10(max(purchases, 1)) +
                           0.20 * (return_rate - thresholds['high_returner_rate']))

            reason = (
                f"Customer {customer['customer_id']} has {return_rate:.0%} return rate "
                f"({int(customer['recent_returns'])}/{int(customer['recent_purchases'])} items) "
                f"in the last {thresholds['period_days']} days. "
                f"Threshold: {thresholds['high_returner_rate']:.0%}. "
                f"Current segment: {customer['current_segment']}."
            )

            supporting_data = {
                'customer_id': customer['customer_id'],
                'current_segment': customer['current_segment'],
                'suggested_segment': 'high_returner',
                'recent_returns': int(customer['recent_returns']),
                'recent_purchases': int(customer['recent_purchases']),
                'recent_return_rate': float(return_rate),
                'recent_orders': int(customer['recent_orders']),
                'period_days': thresholds['period_days'],
            }

            estimated_impact = {
                'return_reduction': 0.0005,
                'cvr_impact': 0.0,
                'net_margin_impact': float(customer['recent_returns'] * self.analyzer.COST_PER_RETURN * 0.15),
            }

            mapp_api_call = {
                'endpoint': 'mapp-intelligence/ci-segments',
                'method': 'PUT',
                'params': {
                    'customer_id': customer['customer_id'],
                    'segment_action': 'add',
                    'segment_name': 'high_returner',
                    'reason': 'automated_resegment',
                    'ttl_days': 90,
                },
            }

            action = Action(
                action_type=ActionType.RESEGMENT,
                target=customer['customer_id'],
                confidence=confidence,
                reason=reason,
                supporting_data=supporting_data,
                estimated_impact=estimated_impact,
                mapp_api_call=mapp_api_call,
            )

            decisions.append(action)

        return decisions

    def get_decisions_summary(self, decisions: List[Action] = None) -> Dict[str, Any]:
        """
        Get a summary of decisions for reporting.

        Args:
            decisions: List of decisions (generates if not provided)

        Returns:
            Summary dict with counts and highlights
        """
        if decisions is None:
            decisions = self.generate_decisions()

        summary = {
            'total_decisions': len(decisions),
            'by_type': {},
            'by_confidence': {
                'high': 0,    # >= 0.8
                'medium': 0,  # >= 0.6
                'low': 0,     # < 0.6
            },
            'total_estimated_margin_impact': 0,
            'high_confidence_decisions': [],
        }

        min_confidence = self.thresholds['general']['min_confidence_to_act']

        for d in decisions:
            # Count by type
            type_name = d.action_type.value
            if type_name not in summary['by_type']:
                summary['by_type'][type_name] = 0
            summary['by_type'][type_name] += 1

            # Count by confidence
            if d.confidence >= 0.8:
                summary['by_confidence']['high'] += 1
            elif d.confidence >= 0.6:
                summary['by_confidence']['medium'] += 1
            else:
                summary['by_confidence']['low'] += 1

            # Sum margin impact
            summary['total_estimated_margin_impact'] += d.estimated_impact.get('net_margin_impact', 0)

            # Track high confidence decisions
            if d.confidence >= min_confidence:
                summary['high_confidence_decisions'].append({
                    'type': type_name,
                    'target': d.target,
                    'confidence': d.confidence,
                    'reason': d.reason[:100] + '...' if len(d.reason) > 100 else d.reason,
                })

        # Limit high confidence list
        summary['high_confidence_decisions'] = summary['high_confidence_decisions'][:10]

        return summary
