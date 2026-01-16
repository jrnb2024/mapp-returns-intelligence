"""
Impact Calculator for the Returns Intelligence Agent.

Provides sophisticated impact estimation for decisions, including
confidence intervals, break-even analysis, and portfolio optimization.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from src.analysis_engine import ReturnsAnalyzer
from src.actions import Action, ActionType


class ImpactCalculator:
    """
    Calculates detailed impact estimates for decisions.

    Uses historical data and conservative assumptions to project:
    - Returns prevented
    - Cost savings
    - Revenue at risk
    - Net margin impact
    - Confidence intervals

    Methodology notes:
    - Uses conservative estimates (lower bound of expected impact)
    - Confidence intervals based on sample size and variance
    - Break-even analysis identifies decision thresholds
    """

    # Cost assumptions (should match ReturnsAnalyzer)
    COST_PER_RETURN = 6.00  # £ per returned item (processing, shipping, restocking)
    GROSS_MARGIN = 0.55  # 55% gross margin
    CVR_IMPACT_SUPPRESS = -0.02  # 2% CVR reduction from suppressing products
    CVR_IMPACT_WARN = -0.05  # 5% CVR reduction from warnings (friction)

    # Effectiveness assumptions (conservative)
    SUPPRESS_EFFECTIVENESS = 0.70  # 70% of excess returns prevented
    WARN_EFFECTIVENESS = 0.15  # 15% of returns prevented by warnings
    RESEGMENT_EFFECTIVENESS = 0.20  # 20% return reduction from treatment changes

    def __init__(self, analyzer: ReturnsAnalyzer):
        """
        Initialize with analyzer for baseline metrics.

        Args:
            analyzer: ReturnsAnalyzer instance with transaction data
        """
        self.analyzer = analyzer
        self._cache_baseline_metrics()

    def _cache_baseline_metrics(self) -> None:
        """Cache baseline metrics for impact calculations."""
        summary = self.analyzer.get_executive_summary()

        self.total_items = summary['total_items_sold']
        self.total_returns = summary['total_items_returned']
        self.overall_return_rate = summary['return_rate_qty']
        self.avg_item_value = summary['avg_item_value']
        self.total_revenue = summary['total_gross_revenue']

    def calculate_suppress_impact(self, decision: Action) -> Dict[str, Any]:
        """
        Calculate detailed impact of suppressing a product.

        Suppression removes a product from recommendations, reducing its
        visibility and sales. The trade-off is between return cost savings
        and lost revenue.

        Methodology:
        - Estimate returns prevented based on excess return rate
        - Calculate cost savings from prevented returns
        - Estimate revenue loss from reduced visibility
        - Compute net margin impact

        Args:
            decision: The SUPPRESS Action to analyze

        Returns:
            Dictionary with detailed impact metrics
        """
        data = decision.supporting_data

        volume = data.get('volume', 0)
        return_rate = data.get('return_rate', 0)
        category_baseline = data.get('category_baseline', self.overall_return_rate)
        excess_return_rate = data.get('excess_return_rate', return_rate - category_baseline)
        avg_price = self.avg_item_value  # Could be product-specific

        # Calculate returns prevented
        # Assumption: Suppression prevents SUPPRESS_EFFECTIVENESS of excess returns
        excess_returns = volume * max(0, excess_return_rate)
        returns_prevented = excess_returns * self.SUPPRESS_EFFECTIVENESS

        # Cost savings from prevented returns
        return_cost_saved = returns_prevented * self.COST_PER_RETURN

        # Revenue at risk (suppression reduces sales)
        # Assumption: 30% of future sales come from recommendations
        recommendation_contribution = 0.30
        estimated_future_sales = volume * 0.5  # Next 6 months projection
        sales_at_risk = estimated_future_sales * recommendation_contribution
        revenue_at_risk = sales_at_risk * avg_price

        # Margin lost from reduced sales
        margin_lost = revenue_at_risk * self.GROSS_MARGIN

        # CVR impact (slight decrease in category conversion)
        cvr_impact = self.CVR_IMPACT_SUPPRESS

        # Net margin impact
        net_margin_impact = return_cost_saved - margin_lost

        # Confidence interval based on sample size
        # Larger samples = tighter confidence interval
        confidence_factor = min(1.0, np.sqrt(volume / 100))
        ci_lower = net_margin_impact * (0.5 + 0.3 * confidence_factor)
        ci_upper = net_margin_impact * (1.2 + 0.3 * confidence_factor)

        # Break-even analysis
        # At what return rate does suppression become neutral?
        if volume > 0 and avg_price > 0:
            # Margin from suppression = Cost saved - Margin lost
            # 0 = (volume * excess_rate * effectiveness * cost) - (sales_risk * margin)
            # Solve for excess_rate
            margin_loss_per_unit = recommendation_contribution * 0.5 * avg_price * self.GROSS_MARGIN
            cost_save_per_unit = self.SUPPRESS_EFFECTIVENESS * self.COST_PER_RETURN
            break_even_excess = margin_loss_per_unit / cost_save_per_unit if cost_save_per_unit > 0 else 0
            break_even_return_rate = category_baseline + break_even_excess
        else:
            break_even_return_rate = 0.5

        return {
            'action_type': 'SUPPRESS',
            'target': decision.target,
            'returns_prevented': round(returns_prevented, 1),
            'return_cost_saved': round(return_cost_saved, 2),
            'revenue_at_risk': round(revenue_at_risk, 2),
            'margin_lost': round(margin_lost, 2),
            'cvr_impact': cvr_impact,
            'net_margin_impact': round(net_margin_impact, 2),
            'confidence_interval': [round(ci_lower, 2), round(ci_upper, 2)],
            'break_even_return_rate': round(break_even_return_rate, 3),
            'recommendation': self._get_suppress_recommendation(net_margin_impact, decision.confidence),
            'assumptions': {
                'effectiveness': self.SUPPRESS_EFFECTIVENESS,
                'recommendation_contribution': recommendation_contribution,
                'cost_per_return': self.COST_PER_RETURN,
            },
        }

    def calculate_warn_impact(self, decision: Action) -> Dict[str, Any]:
        """
        Calculate impact of adding a sizing warning.

        Warnings reduce returns by helping customers choose correct sizes,
        but add friction that may reduce conversion rate.

        Methodology:
        - Estimate customers helped to choose correct size
        - Calculate returns prevented
        - Estimate CVR reduction from friction
        - Compute net margin impact

        Args:
            decision: The WARN Action to analyze

        Returns:
            Dictionary with detailed impact metrics
        """
        data = decision.supporting_data

        volume = data.get('volume', 0)
        return_rate = data.get('return_rate', 0)
        sizing_issue_rate = data.get('combined_sizing_issue', 0.3)
        avg_price = self.avg_item_value

        # Estimate sizing-related returns
        total_returns = volume * return_rate
        sizing_returns = total_returns * sizing_issue_rate

        # Returns prevented by warning
        # Assumption: WARN_EFFECTIVENESS of sizing returns are prevented
        returns_prevented = sizing_returns * self.WARN_EFFECTIVENESS

        # Cost savings
        return_cost_saved = returns_prevented * self.COST_PER_RETURN

        # Customers helped (ordered correct size instead of wrong)
        # Some portion would have returned, others would have kept wrong size
        customers_helped = int(sizing_returns * 0.5)  # Half of sizing issues

        # CVR impact (warning adds friction)
        cvr_impact = self.CVR_IMPACT_WARN

        # Revenue impact from CVR reduction
        # Estimate future traffic and conversion
        estimated_views = volume * 3  # 3x views per purchase
        lost_conversions = estimated_views * abs(cvr_impact) * 0.1  # 10% of CVR impact
        revenue_lost = lost_conversions * avg_price
        margin_lost = revenue_lost * self.GROSS_MARGIN

        # Net margin impact
        net_margin_impact = return_cost_saved - margin_lost

        # Confidence interval
        confidence_factor = min(1.0, np.sqrt(volume / 50))
        ci_lower = net_margin_impact * (0.4 + 0.3 * confidence_factor)
        ci_upper = net_margin_impact * (1.5 + 0.3 * confidence_factor)

        # Determine targeting recommendation
        if sizing_issue_rate > 0.5:
            targeting = 'Show to all customers'
        elif sizing_issue_rate > 0.35:
            targeting = 'Show to new customers only'
        else:
            targeting = 'Show only when customer views size guide'

        return {
            'action_type': 'WARN',
            'target': decision.target,
            'returns_prevented': round(returns_prevented, 1),
            'return_cost_saved': round(return_cost_saved, 2),
            'customers_helped': customers_helped,
            'cvr_impact': cvr_impact,
            'revenue_at_risk': round(revenue_lost, 2),
            'margin_lost': round(margin_lost, 2),
            'net_margin_impact': round(net_margin_impact, 2),
            'confidence_interval': [round(ci_lower, 2), round(ci_upper, 2)],
            'targeting_recommendation': targeting,
            'recommendation': self._get_warn_recommendation(net_margin_impact, sizing_issue_rate),
            'assumptions': {
                'effectiveness': self.WARN_EFFECTIVENESS,
                'cvr_reduction': self.CVR_IMPACT_WARN,
                'cost_per_return': self.COST_PER_RETURN,
            },
        }

    def calculate_resegment_impact(self, decision: Action) -> Dict[str, Any]:
        """
        Calculate impact of moving customer to different segment.

        Resegmentation changes how we treat a customer, potentially
        reducing their return rate but also risking reduced engagement.

        Methodology:
        - Estimate return reduction from treatment changes
        - Calculate potential order reduction
        - Assess churn risk
        - Compute net LTV impact

        Args:
            decision: The RESEGMENT Action to analyze

        Returns:
            Dictionary with detailed impact metrics
        """
        data = decision.supporting_data

        current_segment = data.get('current_segment', 'normal_returner')
        suggested_segment = data.get('suggested_segment', 'high_returner')
        recent_returns = data.get('recent_returns', 0)
        recent_purchases = data.get('recent_purchases', 0)
        recent_return_rate = data.get('recent_return_rate', 0)

        # Estimate customer metrics
        avg_order_value = self.avg_item_value * 2.5  # Average items per order
        orders_per_year = max(4, recent_purchases / 3 * 4)  # Annualized

        # Treatment change description
        treatment_changes = {
            'serial_returner': 'Reduced recommendation frequency, limited promotions',
            'high_returner': 'Conservative recommendations, standard treatment',
        }
        treatment_change = treatment_changes.get(suggested_segment, 'Modified treatment')

        # Expected return reduction
        # Moving to stricter segment changes recommendation strategy
        expected_return_reduction = self.RESEGMENT_EFFECTIVENESS

        # Returns prevented per year
        annual_returns = orders_per_year * recent_return_rate * 2.5
        returns_prevented = annual_returns * expected_return_reduction

        # Cost savings
        return_cost_saved = returns_prevented * self.COST_PER_RETURN

        # Expected order reduction (customers may disengage)
        # Serial returners get less engagement, may order less
        if suggested_segment == 'serial_returner':
            expected_order_reduction = 0.15  # 15% fewer orders
            churn_risk = 0.12  # 12% chance of churning
        else:
            expected_order_reduction = 0.08  # 8% fewer orders
            churn_risk = 0.06  # 6% chance of churning

        # Revenue impact from reduced orders
        orders_lost = orders_per_year * expected_order_reduction
        revenue_lost = orders_lost * avg_order_value * (1 - recent_return_rate)  # Net of returns
        margin_lost = revenue_lost * self.GROSS_MARGIN

        # Net LTV impact (annual)
        net_ltv_impact = return_cost_saved - margin_lost

        # Confidence based on customer history
        confidence_factor = min(1.0, np.sqrt(recent_purchases / 10))
        ci_lower = net_ltv_impact * (0.3 + 0.4 * confidence_factor)
        ci_upper = net_ltv_impact * (1.4 + 0.3 * confidence_factor)

        return {
            'action_type': 'RESEGMENT',
            'target': decision.target,
            'treatment_change': treatment_change,
            'expected_return_reduction': expected_return_reduction,
            'returns_prevented_annual': round(returns_prevented, 1),
            'return_cost_saved_annual': round(return_cost_saved, 2),
            'expected_order_reduction': expected_order_reduction,
            'orders_lost_annual': round(orders_lost, 1),
            'revenue_at_risk_annual': round(revenue_lost, 2),
            'margin_lost_annual': round(margin_lost, 2),
            'net_ltv_impact': round(net_ltv_impact, 2),
            'confidence_interval': [round(ci_lower, 2), round(ci_upper, 2)],
            'churn_risk': churn_risk,
            'recommendation': self._get_resegment_recommendation(net_ltv_impact, churn_risk),
            'assumptions': {
                'effectiveness': self.RESEGMENT_EFFECTIVENESS,
                'cost_per_return': self.COST_PER_RETURN,
            },
        }

    def calculate_portfolio_impact(self, decisions: List[Action]) -> Dict[str, Any]:
        """
        Calculate combined impact of all decisions.

        Aggregates individual impacts and identifies:
        - Total returns prevented
        - Net margin impact
        - Return rate reduction
        - Execution priority recommendations

        Args:
            decisions: List of Actions to analyze

        Returns:
            Dictionary with portfolio-level impact metrics
        """
        # Initialize aggregates
        total_returns_prevented = 0.0
        total_return_cost_saved = 0.0
        total_revenue_at_risk = 0.0
        total_margin_lost = 0.0
        total_net_margin_impact = 0.0

        by_type = {
            'SUPPRESS': {'count': 0, 'returns_prevented': 0, 'net_impact': 0},
            'WARN': {'count': 0, 'returns_prevented': 0, 'net_impact': 0},
            'RESEGMENT': {'count': 0, 'returns_prevented': 0, 'net_impact': 0},
        }

        high_confidence_decisions = []
        negative_impact_decisions = []

        for decision in decisions:
            # Calculate individual impact
            if decision.action_type == ActionType.SUPPRESS:
                impact = self.calculate_suppress_impact(decision)
            elif decision.action_type == ActionType.WARN:
                impact = self.calculate_warn_impact(decision)
            elif decision.action_type == ActionType.RESEGMENT:
                impact = self.calculate_resegment_impact(decision)
            else:
                continue

            # Aggregate
            returns_prevented = impact.get('returns_prevented', impact.get('returns_prevented_annual', 0))
            return_cost_saved = impact.get('return_cost_saved', impact.get('return_cost_saved_annual', 0))
            revenue_at_risk = impact.get('revenue_at_risk', impact.get('revenue_at_risk_annual', 0))
            margin_lost = impact.get('margin_lost', impact.get('margin_lost_annual', 0))
            net_impact = impact.get('net_margin_impact', impact.get('net_ltv_impact', 0))

            total_returns_prevented += returns_prevented
            total_return_cost_saved += return_cost_saved
            total_revenue_at_risk += revenue_at_risk
            total_margin_lost += margin_lost
            total_net_margin_impact += net_impact

            # Track by type
            action_type = decision.action_type.value
            by_type[action_type]['count'] += 1
            by_type[action_type]['returns_prevented'] += returns_prevented
            by_type[action_type]['net_impact'] += net_impact

            # Track high confidence positive decisions
            if decision.confidence >= 0.7 and net_impact > 0:
                high_confidence_decisions.append({
                    'target': decision.target,
                    'type': action_type,
                    'confidence': decision.confidence,
                    'net_impact': net_impact,
                })

            # Track negative impact decisions
            if net_impact < 0:
                negative_impact_decisions.append({
                    'target': decision.target,
                    'type': action_type,
                    'net_impact': net_impact,
                })

        # Calculate return rate reduction
        if self.total_returns > 0:
            return_rate_reduction = total_returns_prevented / self.total_returns
        else:
            return_rate_reduction = 0

        # Determine confidence level
        if len(high_confidence_decisions) > len(decisions) * 0.5:
            confidence = 'high'
        elif len(high_confidence_decisions) > len(decisions) * 0.25:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Generate recommendation
        recommendation = self._get_portfolio_recommendation(
            by_type, high_confidence_decisions, negative_impact_decisions
        )

        # Sort high confidence by impact
        high_confidence_decisions.sort(key=lambda x: x['net_impact'], reverse=True)

        return {
            'total_decisions': len(decisions),
            'total_returns_prevented': round(total_returns_prevented, 0),
            'total_return_cost_saved': round(total_return_cost_saved, 2),
            'total_revenue_at_risk': round(total_revenue_at_risk, 2),
            'total_margin_lost': round(total_margin_lost, 2),
            'net_margin_impact': round(total_net_margin_impact, 2),
            'return_rate_reduction': round(return_rate_reduction, 4),
            'return_rate_reduction_pct': f"{return_rate_reduction:.2%}",
            'by_type': by_type,
            'confidence': confidence,
            'high_confidence_decisions': high_confidence_decisions[:10],
            'negative_impact_decisions': negative_impact_decisions[:5],
            'recommendation': recommendation,
            'execution_priority': self._get_execution_priority(by_type),
        }

    def _get_suppress_recommendation(self, net_impact: float, confidence: float) -> str:
        """Generate recommendation for SUPPRESS decision."""
        if net_impact > 50 and confidence >= 0.7:
            return "Strong recommend: High-confidence positive impact"
        elif net_impact > 0 and confidence >= 0.6:
            return "Recommend: Positive expected impact"
        elif net_impact > 0:
            return "Consider: Positive but lower confidence"
        else:
            return "Review: May have negative net impact"

    def _get_warn_recommendation(self, net_impact: float, sizing_issue_rate: float) -> str:
        """Generate recommendation for WARN decision."""
        if net_impact > 20 and sizing_issue_rate > 0.4:
            return "Strong recommend: Clear sizing issue, positive impact"
        elif net_impact > 0:
            return "Recommend: Expected positive impact"
        elif sizing_issue_rate > 0.5:
            return "Consider A/B test: High sizing issue but marginal impact"
        else:
            return "Defer: Impact uncertain, consider monitoring first"

    def _get_resegment_recommendation(self, net_ltv_impact: float, churn_risk: float) -> str:
        """Generate recommendation for RESEGMENT decision."""
        if net_ltv_impact > 0 and churn_risk < 0.1:
            return "Recommend: Positive LTV impact with low churn risk"
        elif net_ltv_impact > 0:
            return "Consider: Positive impact but monitor for churn"
        elif churn_risk > 0.15:
            return "Caution: High churn risk may outweigh benefits"
        else:
            return "Defer: Marginal impact, continue monitoring"

    def _get_portfolio_recommendation(
        self,
        by_type: Dict,
        high_confidence: List,
        negative_impact: List,
    ) -> str:
        """Generate portfolio-level recommendation."""
        recommendations = []

        # SUPPRESS recommendations
        if by_type['SUPPRESS']['count'] > 0:
            avg_impact = by_type['SUPPRESS']['net_impact'] / by_type['SUPPRESS']['count']
            if avg_impact > 10:
                recommendations.append(
                    f"Execute {by_type['SUPPRESS']['count']} SUPPRESS actions (avg £{avg_impact:.0f} each)"
                )

        # WARN recommendations
        if by_type['WARN']['count'] > 0:
            recommendations.append(
                f"Deploy {by_type['WARN']['count']} sizing warnings via A/B test"
            )

        # RESEGMENT recommendations
        if by_type['RESEGMENT']['count'] > 0:
            recommendations.append(
                f"Batch update {by_type['RESEGMENT']['count']} customer segments"
            )

        # Warnings
        if len(negative_impact) > 5:
            recommendations.append(
                f"Review {len(negative_impact)} decisions with negative expected impact"
            )

        return "; ".join(recommendations) if recommendations else "Review individual decisions"

    def _get_execution_priority(self, by_type: Dict) -> List[str]:
        """Determine execution priority based on impact."""
        priorities = []

        # Calculate impact per decision for each type
        type_efficiency = []
        for action_type, data in by_type.items():
            if data['count'] > 0:
                efficiency = data['net_impact'] / data['count']
                type_efficiency.append((action_type, efficiency, data['count']))

        # Sort by efficiency
        type_efficiency.sort(key=lambda x: x[1], reverse=True)

        for i, (action_type, efficiency, count) in enumerate(type_efficiency, 1):
            if efficiency > 0:
                priorities.append(
                    f"{i}. {action_type}: {count} decisions (avg £{efficiency:.0f}/decision)"
                )

        return priorities if priorities else ["No positive-impact decisions identified"]
