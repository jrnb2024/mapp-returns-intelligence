"""
Returns Analysis Engine for Mapp Fashion

Computes all metrics matching Sarah's Hush Returns Consultancy:
- Executive Summary with £ impact
- Category benchmarking
- Size analysis
- HVHR product identification
- Attribute drivers
- Customer segmentation
- Multi-buy analysis
- Time trends
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any


class ReturnsAnalyzer:
    """
    Comprehensive returns analysis engine.
    Produces insights matching Sarah's Hush Returns Consultancy report.
    """
    
    # Business assumptions (configurable per retailer)
    COST_PER_RETURN = 1.08  # £ per returned item
    GROSS_MARGIN = 0.685  # 68.5%
    DISTRIBUTION_COST_PER_ORDER = 4.19  # £
    
    # Industry benchmarks by category
    INDUSTRY_BENCHMARKS = {
        'Dresses': 0.45,
        'Jeans': 0.50,
        'Tops': 0.28,
        'Jumpers': 0.32,
        'Skirts': 0.40,
        'Outerwear': 0.45,
        'Shorts': 0.38,
        'Jumpsuits': 0.48,
        'Sweatshirts': 0.30,
        'Pyjamas': 0.22,
        'Accessories': 0.12,
    }
    
    def __init__(self, transactions_df: pd.DataFrame, products_df: pd.DataFrame = None, 
                 customers_df: pd.DataFrame = None):
        """Initialize with transaction data."""
        self.transactions = transactions_df.copy()
        self.products = products_df.copy() if products_df is not None else None
        self.customers = customers_df.copy() if customers_df is not None else None
        
        # Ensure date column is datetime
        if 'order_date' in self.transactions.columns:
            self.transactions['order_date'] = pd.to_datetime(self.transactions['order_date'])
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary with key metrics and financial impact.
        Matches slide 4 of Sarah's report.
        """
        total_qty = self.transactions['qty_purchased'].sum()
        total_returns = self.transactions['qty_returned'].sum()
        total_value = self.transactions['value_purchased'].sum()
        total_return_value = self.transactions['value_returned'].sum()
        
        return_rate_qty = total_returns / total_qty
        return_rate_value = total_return_value / total_value
        
        # Financial impact calculations
        cost_of_returns = total_returns * self.COST_PER_RETURN
        
        # Value of 1% reduction in returns
        items_per_pct = total_qty * 0.01
        net_revenue_per_pct = items_per_pct * (total_value / total_qty)
        ebit_per_pct = net_revenue_per_pct * self.GROSS_MARGIN
        
        # Net revenue (after returns)
        net_revenue = total_value - total_return_value
        
        return {
            'total_items_sold': int(total_qty),
            'total_items_returned': int(total_returns),
            'total_gross_revenue': total_value,
            'total_return_value': total_return_value,
            'net_revenue': net_revenue,
            'return_rate_qty': return_rate_qty,
            'return_rate_value': return_rate_value,
            'cost_of_returns': cost_of_returns,
            'value_per_1pct_reduction_revenue': net_revenue_per_pct,
            'value_per_1pct_reduction_ebit': ebit_per_pct,
            'avg_item_value': total_value / total_qty,
        }
    
    # =========================================================================
    # CATEGORY ANALYSIS
    # =========================================================================
    
    def get_category_analysis(self) -> pd.DataFrame:
        """
        Analyze return rates by category with benchmarking.
        Matches slides 12-14 of Sarah's report.
        """
        cat_stats = self.transactions.groupby('category').agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'value_purchased': 'sum',
            'value_returned': 'sum',
            'order_id': 'nunique',
        }).rename(columns={'order_id': 'num_orders'})
        
        cat_stats['return_rate'] = cat_stats['qty_returned'] / cat_stats['qty_purchased']
        cat_stats['avg_value'] = cat_stats['value_purchased'] / cat_stats['qty_purchased']
        cat_stats['pct_of_sales'] = cat_stats['qty_purchased'] / cat_stats['qty_purchased'].sum()
        cat_stats['pct_of_returns'] = cat_stats['qty_returned'] / cat_stats['qty_returned'].sum()
        
        # Add industry benchmark comparison
        cat_stats['industry_benchmark'] = cat_stats.index.map(
            lambda x: self.INDUSTRY_BENCHMARKS.get(x, 0.35)
        )
        cat_stats['vs_benchmark'] = cat_stats['return_rate'] - cat_stats['industry_benchmark']
        
        # Calculate contribution to overall return rate
        overall_rate = cat_stats['qty_returned'].sum() / cat_stats['qty_purchased'].sum()
        cat_stats['rate_contribution'] = (
            cat_stats['pct_of_sales'] * (cat_stats['return_rate'] - overall_rate)
        )
        
        return cat_stats.sort_values('return_rate', ascending=False)
    
    # =========================================================================
    # SIZE ANALYSIS
    # =========================================================================
    
    def get_size_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze return rates by size.
        Matches slides 16-18 of Sarah's report.
        """
        # Filter out non-sized items
        sized = self.transactions[self.transactions['size'] != 'One Size'].copy()
        
        # Numeric sizes
        numeric_sizes = ['4', '6', '8', '10', '12', '14', '16', '18']
        numeric_df = sized[sized['size'].isin(numeric_sizes)]
        
        if len(numeric_df) > 0:
            numeric_stats = numeric_df.groupby('size').agg({
                'qty_purchased': 'sum',
                'qty_returned': 'sum',
            })
            numeric_stats['return_rate'] = numeric_stats['qty_returned'] / numeric_stats['qty_purchased']
            # Reindex to maintain order
            numeric_stats = numeric_stats.reindex(numeric_sizes).dropna()
        else:
            numeric_stats = pd.DataFrame()
        
        # Alpha sizes
        alpha_sizes = ['XS', 'S', 'M', 'L', 'XL']
        alpha_df = sized[sized['size'].isin(alpha_sizes)]
        
        if len(alpha_df) > 0:
            alpha_stats = alpha_df.groupby('size').agg({
                'qty_purchased': 'sum',
                'qty_returned': 'sum',
            })
            alpha_stats['return_rate'] = alpha_stats['qty_returned'] / alpha_stats['qty_purchased']
            alpha_stats = alpha_stats.reindex(alpha_sizes).dropna()
        else:
            alpha_stats = pd.DataFrame()
        
        # Calculate size vs overall baseline
        overall_rate = sized['qty_returned'].sum() / sized['qty_purchased'].sum()
        if len(numeric_stats) > 0:
            numeric_stats['vs_baseline'] = numeric_stats['return_rate'] - overall_rate
        if len(alpha_stats) > 0:
            alpha_stats['vs_baseline'] = alpha_stats['return_rate'] - overall_rate
        
        return {
            'numeric': numeric_stats,
            'alpha': alpha_stats,
            'overall_rate': overall_rate,
        }
    
    def get_size_keeping_patterns(self) -> pd.DataFrame:
        """
        Analyze which size customers keep when ordering multiple sizes.
        Matches slide 17 of Sarah's report.
        """
        # Find multi-size orders
        multi_size = self.transactions[self.transactions['is_multi_size_order'] == True].copy()
        
        if len(multi_size) == 0:
            return pd.DataFrame()
        
        # Group by order and product to find size pairs
        size_pairs = multi_size.groupby(['order_id', 'product_id']).apply(
            lambda x: pd.Series({
                'sizes': sorted(x['size'].tolist()),
                'returned_sizes': x[x['qty_returned'] > 0]['size'].tolist(),
                'kept_sizes': x[x['qty_returned'] == 0]['size'].tolist(),
            })
        ).reset_index()
        
        # Analyze keeping patterns
        size_pairs['kept_smaller'] = size_pairs.apply(
            lambda row: len(row['kept_sizes']) > 0 and 
                       (len(row['returned_sizes']) == 0 or 
                        min(row['kept_sizes']) < min(row['returned_sizes']) if row['returned_sizes'] else True),
            axis=1
        )
        
        return size_pairs
    
    # =========================================================================
    # HVHR (HIGH VOLUME HIGH RETURN) PRODUCTS
    # =========================================================================
    
    def get_hvhr_products(self, min_qty: int = 50, top_n: int = 50) -> pd.DataFrame:
        """
        Identify High Volume, High Return products.
        Matches slides 41-43 of Sarah's report.
        """
        # Aggregate by product
        product_stats = self.transactions.groupby(['product_id', 'product_name', 'category']).agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'value_purchased': 'sum',
            'value_returned': 'sum',
        }).reset_index()
        
        # Calculate return rate
        product_stats['return_rate'] = product_stats['qty_returned'] / product_stats['qty_purchased']
        
        # Get category baselines
        category_baseline = self.transactions.groupby('category').apply(
            lambda x: x['qty_returned'].sum() / x['qty_purchased'].sum()
        ).to_dict()
        
        product_stats['category_baseline'] = product_stats['category'].map(category_baseline)
        product_stats['excess_return_rate'] = product_stats['return_rate'] - product_stats['category_baseline']
        
        # Filter for high volume
        hvhr = product_stats[product_stats['qty_purchased'] >= min_qty].copy()
        
        # Score: excess return rate * log(volume)
        hvhr['hvhr_score'] = hvhr['excess_return_rate'].clip(lower=0) * np.log1p(hvhr['qty_purchased'])
        
        # Get top HVHR products
        hvhr = hvhr.nlargest(top_n, 'hvhr_score')
        
        # Add financial impact
        overall_baseline = self.transactions['qty_returned'].sum() / self.transactions['qty_purchased'].sum()
        hvhr['excess_returns'] = hvhr['qty_purchased'] * hvhr['excess_return_rate'].clip(lower=0)
        hvhr['cost_impact'] = hvhr['excess_returns'] * self.COST_PER_RETURN
        
        return hvhr
    
    # =========================================================================
    # ATTRIBUTE DRIVERS
    # =========================================================================
    
    def get_attribute_drivers(self, min_sample_size: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Analyze which product attributes drive higher return rates.
        Matches slides 44-45 of Sarah's report.
        """
        attribute_columns = ['fit', 'neckline', 'pattern', 'sleeve_style', 'length', 
                           'occasion', 'style_aesthetic', 'trend']
        
        # Filter to columns that exist
        attribute_columns = [c for c in attribute_columns if c in self.transactions.columns]
        
        overall_rate = self.transactions['qty_returned'].sum() / self.transactions['qty_purchased'].sum()
        
        results = {}
        
        for attr in attribute_columns:
            # Skip empty/null values
            valid = self.transactions[self.transactions[attr].notna() & (self.transactions[attr] != '-')]
            
            if len(valid) == 0:
                continue
            
            attr_stats = valid.groupby(attr).agg({
                'qty_purchased': 'sum',
                'qty_returned': 'sum',
                'product_id': 'nunique',
            }).rename(columns={'product_id': 'n_products'})
            
            attr_stats['return_rate'] = attr_stats['qty_returned'] / attr_stats['qty_purchased']
            attr_stats['lift'] = attr_stats['return_rate'] - overall_rate
            attr_stats['pct_of_sales'] = attr_stats['qty_purchased'] / attr_stats['qty_purchased'].sum()
            
            # Filter for minimum sample size
            attr_stats = attr_stats[attr_stats['qty_purchased'] >= min_sample_size]
            
            if len(attr_stats) > 0:
                results[attr] = attr_stats.sort_values('lift', ascending=False)
        
        return results
    
    # =========================================================================
    # CUSTOMER ANALYSIS
    # =========================================================================
    
    def get_customer_analysis(self) -> Dict[str, Any]:
        """
        Analyze customer return behavior and profitability.
        Matches slides 33-38 of Sarah's report.
        """
        # Aggregate by customer
        customer_stats = self.transactions.groupby('customer_id').agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'value_purchased': 'sum',
            'value_returned': 'sum',
            'order_id': 'nunique',
            'customer_segment': 'first',  # Assuming one segment per customer
        }).rename(columns={'order_id': 'num_orders'})
        
        customer_stats['return_rate'] = customer_stats['qty_returned'] / customer_stats['qty_purchased']
        customer_stats['net_revenue'] = customer_stats['value_purchased'] - customer_stats['value_returned']
        customer_stats['return_cost'] = customer_stats['qty_returned'] * self.COST_PER_RETURN
        customer_stats['profit'] = (customer_stats['net_revenue'] * self.GROSS_MARGIN) - customer_stats['return_cost']
        
        # Classify profitability
        customer_stats['profit_group'] = 'Profitable'
        customer_stats.loc[customer_stats['return_rate'] >= 0.999, 'profit_group'] = 'Returned Everything'
        customer_stats.loc[(customer_stats['profit'] < 0) & (customer_stats['return_rate'] < 0.999), 'profit_group'] = 'Unprofitable'
        
        # Top returners analysis (10% of customers = X% of returns)
        customer_stats_sorted = customer_stats.sort_values('qty_returned', ascending=False)
        total_returns = customer_stats['qty_returned'].sum()
        
        top_10_pct_customers = int(len(customer_stats) * 0.10)
        top_10_pct_returns = customer_stats_sorted.head(top_10_pct_customers)['qty_returned'].sum()
        top_10_pct_of_returns = top_10_pct_returns / total_returns
        
        # Segment analysis - reset index first to make customer_id a column
        customer_stats_reset = customer_stats.reset_index()
        segment_stats = customer_stats_reset.groupby('customer_segment').agg({
            'customer_id': 'count',
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'profit': 'sum',
        }).rename(columns={'customer_id': 'num_customers'})
        segment_stats['return_rate'] = segment_stats['qty_returned'] / segment_stats['qty_purchased']
        segment_stats['pct_of_returns'] = segment_stats['qty_returned'] / total_returns
        
        # Never returners
        never_returners = (customer_stats['qty_returned'] == 0).sum()
        pct_never_return = never_returners / len(customer_stats)
        
        return {
            'customer_stats': customer_stats,
            'segment_stats': segment_stats,
            'top_10_pct_returns': top_10_pct_of_returns,
            'pct_never_return': pct_never_return,
            'total_customers': len(customer_stats),
            'profitable_customers': (customer_stats['profit_group'] == 'Profitable').sum(),
            'unprofitable_customers': (customer_stats['profit_group'] == 'Unprofitable').sum(),
        }
    
    # =========================================================================
    # MULTI-BUY ANALYSIS
    # =========================================================================
    
    def get_multi_buy_analysis(self) -> Dict[str, Any]:
        """
        Analyze multi-style and multi-size purchase patterns.
        Matches slides 24-30 of Sarah's report.
        """
        total_returns = self.transactions['qty_returned'].sum()
        
        # Multi-style analysis
        multi_style = self.transactions[self.transactions['is_multi_style_order'] == True]
        multi_style_returns = multi_style['qty_returned'].sum()
        multi_style_pct_of_returns = multi_style_returns / total_returns if total_returns > 0 else 0
        
        # Multi-size analysis
        multi_size = self.transactions[self.transactions['is_multi_size_order'] == True]
        multi_size_returns = multi_size['qty_returned'].sum()
        multi_size_pct_of_returns = multi_size_returns / total_returns if total_returns > 0 else 0
        
        # Other returns
        other_pct = 1 - multi_style_pct_of_returns - multi_size_pct_of_returns
        
        # By category breakdown
        category_multi = self.transactions.groupby('category').agg({
            'qty_returned': 'sum',
            'is_multi_style_order': lambda x: (x & (self.transactions.loc[x.index, 'qty_returned'] > 0)).sum(),
            'is_multi_size_order': lambda x: (x & (self.transactions.loc[x.index, 'qty_returned'] > 0)).sum(),
        })
        
        return {
            'multi_style_pct_of_returns': multi_style_pct_of_returns,
            'multi_size_pct_of_returns': multi_size_pct_of_returns,
            'other_pct_of_returns': other_pct,
            'multi_style_items_sold': len(multi_style),
            'multi_size_items_sold': len(multi_size),
            'multi_style_return_rate': multi_style['qty_returned'].sum() / multi_style['qty_purchased'].sum() if len(multi_style) > 0 else 0,
            'multi_size_return_rate': multi_size['qty_returned'].sum() / multi_size['qty_purchased'].sum() if len(multi_size) > 0 else 0,
        }
    
    # =========================================================================
    # TIME TRENDS
    # =========================================================================
    
    def get_time_trends(self) -> pd.DataFrame:
        """
        Analyze return rate trends over time.
        Matches slides 6-7, 21-22 of Sarah's report.
        """
        if 'order_date' not in self.transactions.columns:
            return pd.DataFrame()
        
        # Monthly trends
        monthly = self.transactions.groupby(
            self.transactions['order_date'].dt.to_period('M')
        ).agg({
            'qty_purchased': 'sum',
            'qty_returned': 'sum',
            'value_purchased': 'sum',
            'value_returned': 'sum',
            'order_id': 'nunique',
        }).rename(columns={'order_id': 'num_orders'})
        
        monthly['return_rate'] = monthly['qty_returned'] / monthly['qty_purchased']
        monthly['avg_order_value'] = monthly['value_purchased'] / monthly['num_orders']
        
        # Calculate rolling average
        monthly['return_rate_rolling'] = monthly['return_rate'].rolling(3, min_periods=1).mean()
        
        # Reset index for plotting
        monthly = monthly.reset_index()
        monthly['month'] = monthly['order_date'].astype(str)
        
        return monthly
    
    # =========================================================================
    # RETURN REASONS ANALYSIS
    # =========================================================================

    def get_return_reasons_analysis(self) -> Dict[str, Any]:
        """
        Analyze return reasons distribution and patterns.
        Provides insights into why customers return items.
        """
        # Filter to returned items only
        returns = self.transactions[self.transactions['qty_returned'] > 0].copy()

        if 'return_reason' not in returns.columns or len(returns) == 0:
            return {'available': False}

        # Overall reason distribution
        reason_counts = returns['return_reason'].value_counts()
        reason_pcts = returns['return_reason'].value_counts(normalize=True)

        overall_stats = pd.DataFrame({
            'count': reason_counts,
            'percentage': reason_pcts,
            'value_returned': returns.groupby('return_reason')['value_returned'].sum(),
        })
        overall_stats['avg_value'] = overall_stats['value_returned'] / overall_stats['count']

        # Reasons by category
        reason_by_category = returns.groupby(['category', 'return_reason']).agg({
            'qty_returned': 'sum',
            'value_returned': 'sum',
        }).reset_index()

        # Calculate percentage within each category
        category_totals = reason_by_category.groupby('category')['qty_returned'].transform('sum')
        reason_by_category['pct_of_category'] = reason_by_category['qty_returned'] / category_totals

        # Pivot for easier visualization
        reason_category_pivot = reason_by_category.pivot_table(
            index='category',
            columns='return_reason',
            values='pct_of_category',
            fill_value=0
        )

        # Reasons by customer segment
        reason_by_segment = returns.groupby(['customer_segment', 'return_reason']).agg({
            'qty_returned': 'sum',
        }).reset_index()

        segment_totals = reason_by_segment.groupby('customer_segment')['qty_returned'].transform('sum')
        reason_by_segment['pct_of_segment'] = reason_by_segment['qty_returned'] / segment_totals

        reason_segment_pivot = reason_by_segment.pivot_table(
            index='customer_segment',
            columns='return_reason',
            values='pct_of_segment',
            fill_value=0
        )

        # Size-related reasons analysis (Too small/Too large by actual size)
        size_reasons = returns[returns['return_reason'].isin(['Too small', 'Too large'])].copy()
        if len(size_reasons) > 0:
            size_reason_by_size = size_reasons.groupby(['size', 'return_reason']).size().unstack(fill_value=0)
            size_reason_by_size['total'] = size_reason_by_size.sum(axis=1)
            if 'Too small' in size_reason_by_size.columns:
                size_reason_by_size['pct_too_small'] = size_reason_by_size['Too small'] / size_reason_by_size['total']
            if 'Too large' in size_reason_by_size.columns:
                size_reason_by_size['pct_too_large'] = size_reason_by_size['Too large'] / size_reason_by_size['total']
        else:
            size_reason_by_size = pd.DataFrame()

        # Monthly trend of reasons
        if 'order_month' in returns.columns:
            reason_trends = returns.groupby(['order_month', 'return_reason']).size().unstack(fill_value=0)
            # Normalize each month
            reason_trends_pct = reason_trends.div(reason_trends.sum(axis=1), axis=0)
        else:
            reason_trends = pd.DataFrame()
            reason_trends_pct = pd.DataFrame()

        # Top reasons by financial impact
        financial_impact = overall_stats.sort_values('value_returned', ascending=False)

        # Actionable sizing issues (items returned for size reasons)
        sizing_issues = returns[returns['return_reason'].isin(['Too small', 'Too large'])]
        sizing_by_fit = sizing_issues.groupby(['fit', 'return_reason']).size().unstack(fill_value=0) if len(sizing_issues) > 0 else pd.DataFrame()

        return {
            'available': True,
            'overall_stats': overall_stats,
            'reason_by_category': reason_category_pivot,
            'reason_by_segment': reason_segment_pivot,
            'size_reason_by_size': size_reason_by_size,
            'reason_trends': reason_trends,
            'reason_trends_pct': reason_trends_pct,
            'financial_impact': financial_impact,
            'sizing_by_fit': sizing_by_fit,
            'total_returns': len(returns),
            'top_reason': reason_counts.index[0] if len(reason_counts) > 0 else None,
            'top_reason_pct': reason_pcts.iloc[0] if len(reason_pcts) > 0 else 0,
        }

    # =========================================================================
    # RECOMMENDATIONS ENGINE
    # =========================================================================

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on analysis.
        Matches slide 56 of Sarah's report.
        """
        recommendations = []
        
        # Get analyses
        exec_summary = self.get_executive_summary()
        category_analysis = self.get_category_analysis()
        size_analysis = self.get_size_analysis()
        hvhr_products = self.get_hvhr_products(min_qty=30, top_n=20)
        attribute_drivers = self.get_attribute_drivers(min_sample_size=50)
        customer_analysis = self.get_customer_analysis()
        multi_buy = self.get_multi_buy_analysis()
        return_reasons = self.get_return_reasons_analysis()
        
        # 1. HVHR Product Recommendations
        if len(hvhr_products) > 0:
            top_hvhr = hvhr_products.head(5)
            for _, product in top_hvhr.iterrows():
                if product['excess_return_rate'] > 0.10:
                    recommendations.append({
                        'type': 'HVHR Product',
                        'priority': 'High',
                        'category': product['category'],
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'metric': f"{product['return_rate']:.0%} return rate ({product['excess_return_rate']:+.0%} vs category)",
                        'action': 'Review PDP content, add fit guidance, or deprioritise in merchandising',
                        'impact': f"£{product['cost_impact']:.0f} excess return costs",
                    })
        
        # 2. Category Recommendations
        worst_categories = category_analysis[category_analysis['vs_benchmark'] > 0.05].head(3)
        for cat, row in worst_categories.iterrows():
            recommendations.append({
                'type': 'Category Issue',
                'priority': 'Medium',
                'category': cat,
                'metric': f"{row['return_rate']:.0%} vs {row['industry_benchmark']:.0%} benchmark",
                'action': f"Deep dive into {cat} attributes driving returns",
                'impact': f"{row['pct_of_returns']:.0%} of all returns",
            })
        
        # 3. Size Recommendations
        if 'numeric' in size_analysis and len(size_analysis['numeric']) > 0:
            small_sizes = size_analysis['numeric'].head(2)
            if len(small_sizes) > 0 and small_sizes['vs_baseline'].max() > 0.03:
                recommendations.append({
                    'type': 'Sizing Issue',
                    'priority': 'High',
                    'category': 'All Categories',
                    'metric': f"Small sizes return {small_sizes['vs_baseline'].max():.0%} more than average",
                    'action': 'Add sizing guidance to PDPs, review size guide accuracy',
                    'impact': 'Potential 1-2% overall return rate reduction',
                })
        
        # 4. Attribute Recommendations
        for attr, df in attribute_drivers.items():
            high_lift = df[df['lift'] > 0.05]
            if len(high_lift) > 0:
                top_attr = high_lift.index[0]
                lift = high_lift.loc[top_attr, 'lift']
                recommendations.append({
                    'type': 'Attribute Driver',
                    'priority': 'Medium',
                    'category': 'All Categories',
                    'metric': f"{attr}={top_attr}: {lift:+.0%} lift vs average",
                    'action': f"Review imagery and copy for {top_attr} items",
                    'impact': f"{high_lift.loc[top_attr, 'pct_of_sales']:.0%} of sales affected",
                })
        
        # 5. Customer Recommendations
        if customer_analysis['top_10_pct_returns'] > 0.50:
            recommendations.append({
                'type': 'Customer Segment',
                'priority': 'Low',
                'category': 'All Categories',
                'metric': f"Top 10% of customers drive {customer_analysis['top_10_pct_returns']:.0%} of returns",
                'action': 'Identify and engage serial returners with personalised experiences',
                'impact': 'Focus on conversion over acquisition for this segment',
            })
        
        # 6. Multi-Buy Recommendations
        if multi_buy['multi_style_pct_of_returns'] > 0.30:
            recommendations.append({
                'type': 'Multi-Style Wardrobing',
                'priority': 'Medium',
                'category': 'All Categories',
                'metric': f"{multi_buy['multi_style_pct_of_returns']:.0%} of returns from multi-style orders",
                'action': 'Improve personalisation to reduce "try-on" behaviour',
                'impact': 'Potential 2-3% return rate reduction',
            })

        # 7. Return Reasons Recommendations
        if return_reasons.get('available', False):
            overall_stats = return_reasons['overall_stats']

            # Sizing-related recommendations (Too small + Too large)
            sizing_pct = 0
            if 'Too small' in overall_stats.index:
                sizing_pct += overall_stats.loc['Too small', 'percentage']
            if 'Too large' in overall_stats.index:
                sizing_pct += overall_stats.loc['Too large', 'percentage']

            if sizing_pct > 0.40:
                recommendations.append({
                    'type': 'Return Reason - Sizing',
                    'priority': 'High',
                    'category': 'All Categories',
                    'metric': f"{sizing_pct:.0%} of returns due to sizing issues (Too small/Too large)",
                    'action': 'Improve size guides, add garment measurements, show model sizes',
                    'impact': 'Addressing sizing could reduce returns by 10-15%',
                })

            # Quality issues
            if 'Quality' in overall_stats.index and overall_stats.loc['Quality', 'percentage'] > 0.10:
                quality_pct = overall_stats.loc['Quality', 'percentage']
                recommendations.append({
                    'type': 'Return Reason - Quality',
                    'priority': 'High',
                    'category': 'All Categories',
                    'metric': f"{quality_pct:.0%} of returns cite quality issues",
                    'action': 'Review QC processes, investigate supplier quality, update product descriptions',
                    'impact': f"£{overall_stats.loc['Quality', 'value_returned']:,.0f} in returns due to quality",
                })

            # Changed mind (wardrobing indicator)
            if 'Changed mind' in overall_stats.index and overall_stats.loc['Changed mind', 'percentage'] > 0.15:
                changed_mind_pct = overall_stats.loc['Changed mind', 'percentage']
                recommendations.append({
                    'type': 'Return Reason - Changed Mind',
                    'priority': 'Medium',
                    'category': 'All Categories',
                    'metric': f"{changed_mind_pct:.0%} of returns are 'Changed mind'",
                    'action': 'Improve product imagery, add video content, enhance PDP information',
                    'impact': 'Better pre-purchase confidence could reduce impulse returns',
                })

            # Colour different
            if 'Colour different' in overall_stats.index and overall_stats.loc['Colour different', 'percentage'] > 0.08:
                colour_pct = overall_stats.loc['Colour different', 'percentage']
                recommendations.append({
                    'type': 'Return Reason - Colour',
                    'priority': 'Medium',
                    'category': 'All Categories',
                    'metric': f"{colour_pct:.0%} of returns cite colour difference",
                    'action': 'Review product photography, ensure colour accuracy, add multiple lighting shots',
                    'impact': 'Accurate colour representation reduces unexpected returns',
                })

        return recommendations


def run_full_analysis(transactions_path: str, products_path: str = None, 
                      customers_path: str = None) -> Dict[str, Any]:
    """
    Run complete analysis and return all results.
    """
    transactions = pd.read_csv(transactions_path)
    products = pd.read_csv(products_path) if products_path else None
    customers = pd.read_csv(customers_path) if customers_path else None
    
    analyzer = ReturnsAnalyzer(transactions, products, customers)
    
    return {
        'executive_summary': analyzer.get_executive_summary(),
        'category_analysis': analyzer.get_category_analysis(),
        'size_analysis': analyzer.get_size_analysis(),
        'hvhr_products': analyzer.get_hvhr_products(),
        'attribute_drivers': analyzer.get_attribute_drivers(),
        'customer_analysis': analyzer.get_customer_analysis(),
        'multi_buy_analysis': analyzer.get_multi_buy_analysis(),
        'time_trends': analyzer.get_time_trends(),
        'return_reasons_analysis': analyzer.get_return_reasons_analysis(),
        'recommendations': analyzer.generate_recommendations(),
    }
