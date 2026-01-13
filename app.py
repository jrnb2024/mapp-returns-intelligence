"""
Mapp Fashion Returns Intelligence Agent
Streamlit Dashboard with Mapp Branding
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from analysis_engine import ReturnsAnalyzer

# =============================================================================
# PAGE CONFIG & BRANDING
# =============================================================================

st.set_page_config(
    page_title="Mapp Fashion | Returns Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapp brand colors
MAPP_PURPLE = "#5B21B6"  # Dark purple
MAPP_PINK = "#EC4899"    # Magenta/pink
MAPP_LIGHT_PURPLE = "#8B5CF6"  # Lighter purple
MAPP_GRAY = "#6B7280"
MAPP_LIGHT_BG = "#F8FAFC"

# Custom CSS for Mapp branding
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {MAPP_LIGHT_BG};
    }}
    
    /* Header styling */
    .mapp-header {{
        text-align: center;
        padding: 1rem 0 2rem 0;
    }}
    
    .mapp-logo {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .mapp-logo-dark {{
        color: #1F2937;
    }}
    
    .mapp-logo-pink {{
        background: linear-gradient(90deg, {MAPP_PINK}, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .mapp-subtitle {{
        color: {MAPP_GRAY};
        font-size: 1.1rem;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }}
    
    .metric-label {{
        color: {MAPP_GRAY};
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .metric-value {{
        color: #1F2937;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }}
    
    .metric-delta {{
        font-size: 0.9rem;
        font-weight: 500;
    }}
    
    .metric-delta-positive {{
        color: #10B981;
    }}
    
    .metric-delta-negative {{
        color: #EF4444;
    }}
    
    /* Section headers */
    .section-header {{
        color: #1F2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {MAPP_PURPLE};
    }}
    
    /* Recommendation cards */
    .rec-card {{
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid {MAPP_PURPLE};
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    
    .rec-card-high {{
        border-left-color: #EF4444;
    }}
    
    .rec-card-medium {{
        border-left-color: #F59E0B;
    }}
    
    .rec-card-low {{
        border-left-color: #10B981;
    }}
    
    .rec-type {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {MAPP_PURPLE};
        margin-bottom: 0.5rem;
    }}
    
    .rec-action {{
        font-size: 1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }}
    
    .rec-metric {{
        font-size: 0.9rem;
        color: {MAPP_GRAY};
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: 1px solid #E5E7EB;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {MAPP_PURPLE} !important;
        color: white !important;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Data tables */
    .dataframe {{
        border: none !important;
    }}
    
    /* Plotly chart container */
    .chart-container {{
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load transaction and product data."""
    data_dir = Path(__file__).parent / 'data'
    
    transactions = pd.read_csv(data_dir / 'transactions.csv')
    products = pd.read_csv(data_dir / 'products.csv')
    customers = pd.read_csv(data_dir / 'customers.csv')
    
    return transactions, products, customers


@st.cache_data
def run_analysis(_transactions, _products, _customers):
    """Run full analysis (cached)."""
    analyzer = ReturnsAnalyzer(_transactions, _products, _customers)

    return {
        'executive_summary': analyzer.get_executive_summary(),
        'category_analysis': analyzer.get_category_analysis(),
        'size_analysis': analyzer.get_size_analysis(),
        'hvhr_products': analyzer.get_hvhr_products(min_qty=30, top_n=50),
        'attribute_drivers': analyzer.get_attribute_drivers(min_sample_size=50),
        'customer_analysis': analyzer.get_customer_analysis(),
        'multi_buy_analysis': analyzer.get_multi_buy_analysis(),
        'time_trends': analyzer.get_time_trends(),
        'return_reasons': analyzer.get_return_reasons_analysis(),
        'recommendations': analyzer.generate_recommendations(),
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value):
    """Format value as currency."""
    if value >= 1_000_000:
        return f"£{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"£{value/1_000:.0f}K"
    else:
        return f"£{value:.0f}"


def format_number(value):
    """Format large numbers."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def create_metric_card(label, value, delta=None, delta_label=None):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        delta_class = "metric-delta-positive" if delta <= 0 else "metric-delta-negative"
        delta_sign = "↓" if delta <= 0 else "↑"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_sign} {abs(delta):.1%} {delta_label or ""}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def create_recommendation_card(rec):
    """Create a styled recommendation card."""
    priority_class = f"rec-card-{rec['priority'].lower()}"
    
    return f"""
    <div class="rec-card {priority_class}">
        <div class="rec-type">{rec['type']} • {rec['priority']} Priority</div>
        <div class="rec-action">{rec['action']}</div>
        <div class="rec-metric">{rec['metric']}</div>
    </div>
    """


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="mapp-header">
        <div class="mapp-logo">
            <span class="mapp-logo-dark">Mapp</span><span class="mapp-logo-pink">Fashion</span>
        </div>
        <div class="mapp-subtitle">Returns Intelligence Agent</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        transactions, products, customers = load_data()
        analysis = run_analysis(transactions, products, customers)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the synthetic data has been generated by running: `python src/generate_synthetic_data.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 1.5rem; font-weight: 700;">
                <span style="color: #1F2937;">Mapp</span><span style="color: {MAPP_PINK};">Fashion</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Data Overview")
        st.metric("Total Transactions", format_number(len(transactions)))
        st.metric("Products", format_number(len(products)))
        st.metric("Customers", format_number(len(customers)))
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Analysis Settings")
        min_volume = st.slider("Min. product volume for HVHR", 10, 100, 30)
        
        st.markdown("---")
        
        st.caption("🔬 Demo Mode: Synthetic data based on Hush Returns Consultancy patterns")
    
    # Main content tabs
    tab_overview, tab_category, tab_size, tab_reasons, tab_hvhr, tab_drivers, tab_customers, tab_actions = st.tabs([
        "📈 Overview",
        "🏷️ Categories",
        "📏 Sizing",
        "📋 Return Reasons",
        "⚠️ HVHR Products",
        "🔍 Attribute Drivers",
        "👥 Customers",
        "✅ Actions"
    ])
    
    # ==========================================================================
    # TAB: OVERVIEW
    # ==========================================================================
    with tab_overview:
        exec_summary = analysis['executive_summary']
        
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Overall Return Rate",
                f"{exec_summary['return_rate_qty']:.1%}",
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                "Items Returned",
                format_number(exec_summary['total_items_returned']),
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Return Value",
                format_currency(exec_summary['total_return_value']),
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "Value per 1% Reduction",
                format_currency(exec_summary['value_per_1pct_reduction_revenue']),
            ), unsafe_allow_html=True)
        
        # Financial impact callout
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {MAPP_PURPLE}, {MAPP_LIGHT_PURPLE}); 
                    color: white; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
            <div style="font-size: 1.1rem; opacity: 0.9;">💡 Financial Impact</div>
            <div style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">
                Every 1% reduction in returns = {format_currency(exec_summary['value_per_1pct_reduction_ebit'])} EBIT
            </div>
            <div style="font-size: 0.95rem; opacity: 0.9;">
                Based on {format_number(exec_summary['total_items_sold'])} items sold with {exec_summary['return_rate_qty']:.1%} return rate
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Time trend chart
        st.markdown('<div class="section-header">Return Rate Trend (2024)</div>', unsafe_allow_html=True)
        
        time_trends = analysis['time_trends']
        if len(time_trends) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_trends['month'],
                y=time_trends['return_rate'],
                mode='lines+markers',
                name='Return Rate',
                line=dict(color=MAPP_PURPLE, width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=time_trends['month'],
                y=time_trends['return_rate_rolling'],
                mode='lines',
                name='3-Month Rolling Avg',
                line=dict(color=MAPP_PINK, width=2, dash='dash')
            ))
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis_tickformat='.0%',
                height=350,
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Multi-buy breakdown
        st.markdown('<div class="section-header">Returns Breakdown</div>', unsafe_allow_html=True)
        
        multi_buy = analysis['multi_buy_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            breakdown_data = pd.DataFrame({
                'Type': ['Multi-Style Orders', 'Multi-Size Orders', 'Single Item Orders'],
                'Percentage': [
                    multi_buy['multi_style_pct_of_returns'],
                    multi_buy['multi_size_pct_of_returns'],
                    multi_buy['other_pct_of_returns']
                ]
            })
            
            fig = px.pie(
                breakdown_data, 
                values='Percentage', 
                names='Type',
                color_discrete_sequence=[MAPP_PURPLE, MAPP_PINK, MAPP_LIGHT_PURPLE],
                hole=0.4
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.2)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top: 0; color: #1F2937;">Multi-Buy Insights</h4>
                <p><strong>{multi_buy['multi_style_pct_of_returns']:.0%}</strong> of returns come from 
                   customers ordering multiple styles (wardrobing behaviour)</p>
                <p><strong>{multi_buy['multi_size_pct_of_returns']:.0%}</strong> of returns come from 
                   customers ordering multiple sizes</p>
                <p style="color: {MAPP_PURPLE}; font-weight: 600; margin-top: 1rem;">
                   💡 Improving size guidance could reduce multi-size returns significantly
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================================================
    # TAB: CATEGORIES
    # ==========================================================================
    with tab_category:
        st.markdown('<div class="section-header">Category Performance vs Industry Benchmark</div>', unsafe_allow_html=True)
        
        cat_analysis = analysis['category_analysis'].reset_index()
        
        # Bar chart: Return rate vs benchmark
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Your Return Rate',
            x=cat_analysis['category'],
            y=cat_analysis['return_rate'],
            marker_color=MAPP_PURPLE
        ))
        
        fig.add_trace(go.Bar(
            name='Industry Benchmark',
            x=cat_analysis['category'],
            y=cat_analysis['industry_benchmark'],
            marker_color=MAPP_LIGHT_PURPLE,
            opacity=0.6
        ))
        
        fig.update_layout(
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis_tickformat='.0%',
            height=400,
            margin=dict(l=40, r=40, t=20, b=80),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category table
        st.markdown('<div class="section-header">Category Details</div>', unsafe_allow_html=True)
        
        cat_display = cat_analysis[['category', 'qty_purchased', 'qty_returned', 'return_rate', 
                                    'industry_benchmark', 'vs_benchmark', 'pct_of_returns']].copy()
        cat_display.columns = ['Category', 'Items Sold', 'Items Returned', 'Return Rate', 
                              'Benchmark', 'vs Benchmark', '% of All Returns']
        
        # Format columns
        cat_display['Return Rate'] = cat_display['Return Rate'].apply(lambda x: f"{x:.1%}")
        cat_display['Benchmark'] = cat_display['Benchmark'].apply(lambda x: f"{x:.1%}")
        cat_display['vs Benchmark'] = cat_display['vs Benchmark'].apply(lambda x: f"{x:+.1%}")
        cat_display['% of All Returns'] = cat_display['% of All Returns'].apply(lambda x: f"{x:.1%}")
        cat_display['Items Sold'] = cat_display['Items Sold'].apply(lambda x: f"{x:,}")
        cat_display['Items Returned'] = cat_display['Items Returned'].apply(lambda x: f"{x:,}")
        
        st.dataframe(cat_display, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # TAB: SIZING
    # ==========================================================================
    with tab_size:
        st.markdown('<div class="section-header">Size Analysis</div>', unsafe_allow_html=True)
        
        size_data = analysis['size_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Numeric Sizes (Dresses, Jeans, Skirts)")
            
            if len(size_data['numeric']) > 0:
                numeric_df = size_data['numeric'].reset_index()
                
                fig = go.Figure()
                
                # Add baseline reference line
                fig.add_hline(
                    y=size_data['overall_rate'], 
                    line_dash="dash", 
                    line_color=MAPP_GRAY,
                    annotation_text=f"Baseline: {size_data['overall_rate']:.1%}"
                )
                
                # Color bars based on above/below baseline
                colors = [MAPP_PINK if r > size_data['overall_rate'] else MAPP_PURPLE 
                         for r in numeric_df['return_rate']]
                
                fig.add_trace(go.Bar(
                    x=numeric_df['size'],
                    y=numeric_df['return_rate'],
                    marker_color=colors,
                    text=numeric_df['return_rate'].apply(lambda x: f"{x:.1%}"),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis_tickformat='.0%',
                    height=350,
                    margin=dict(l=40, r=40, t=20, b=40),
                    showlegend=False,
                    xaxis=dict(showgrid=False, title='Size'),
                    yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Return Rate'),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric size data available")
        
        with col2:
            st.markdown("#### Alpha Sizes (Tops, Jumpers, Outerwear)")
            
            if len(size_data['alpha']) > 0:
                alpha_df = size_data['alpha'].reset_index()
                
                fig = go.Figure()
                
                fig.add_hline(
                    y=size_data['overall_rate'], 
                    line_dash="dash", 
                    line_color=MAPP_GRAY,
                    annotation_text=f"Baseline: {size_data['overall_rate']:.1%}"
                )
                
                colors = [MAPP_PINK if r > size_data['overall_rate'] else MAPP_PURPLE 
                         for r in alpha_df['return_rate']]
                
                fig.add_trace(go.Bar(
                    x=alpha_df['size'],
                    y=alpha_df['return_rate'],
                    marker_color=colors,
                    text=alpha_df['return_rate'].apply(lambda x: f"{x:.1%}"),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis_tickformat='.0%',
                    height=350,
                    margin=dict(l=40, r=40, t=20, b=40),
                    showlegend=False,
                    xaxis=dict(showgrid=False, title='Size'),
                    yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Return Rate'),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alpha size data available")
        
        # Size insights
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; 
                    border: 1px solid #E5E7EB;">
            <h4 style="margin-top: 0; color: {MAPP_PURPLE};">📏 Size Insights</h4>
            <p>Smaller sizes tend to have higher return rates, suggesting potential sizing inconsistency 
               or customer uncertainty about fit.</p>
            <p><strong>Recommendation:</strong> Add size-specific guidance on PDPs for smaller sizes, 
               especially for fitted items.</p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB: RETURN REASONS
    # ==========================================================================
    with tab_reasons:
        st.markdown('<div class="section-header">Return Reasons Analysis</div>', unsafe_allow_html=True)

        return_reasons = analysis['return_reasons']

        if return_reasons.get('available', False):
            overall_stats = return_reasons['overall_stats']

            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(create_metric_card(
                    "Total Returns Analysed",
                    format_number(return_reasons['total_returns']),
                ), unsafe_allow_html=True)

            with col2:
                st.markdown(create_metric_card(
                    "Top Reason",
                    return_reasons['top_reason'],
                ), unsafe_allow_html=True)

            with col3:
                sizing_pct = 0
                if 'Too small' in overall_stats.index:
                    sizing_pct += overall_stats.loc['Too small', 'percentage']
                if 'Too large' in overall_stats.index:
                    sizing_pct += overall_stats.loc['Too large', 'percentage']
                st.markdown(create_metric_card(
                    "Sizing Issues",
                    f"{sizing_pct:.0%}",
                ), unsafe_allow_html=True)

            with col4:
                quality_pct = overall_stats.loc['Quality', 'percentage'] if 'Quality' in overall_stats.index else 0
                st.markdown(create_metric_card(
                    "Quality Issues",
                    f"{quality_pct:.0%}",
                ), unsafe_allow_html=True)

            # Overall reason distribution
            st.markdown('<div class="section-header">Overall Return Reasons Distribution</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart
                reason_df = overall_stats.reset_index()
                reason_df.columns = ['Reason', 'Count', 'Percentage', 'Value', 'Avg Value']

                fig = px.pie(
                    reason_df,
                    values='Count',
                    names='Reason',
                    color_discrete_sequence=[MAPP_PURPLE, MAPP_PINK, MAPP_LIGHT_PURPLE,
                                            '#F472B6', '#A78BFA', '#6366F1', '#818CF8'],
                    hole=0.4
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Bar chart by value
                reason_df_sorted = reason_df.sort_values('Value', ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=reason_df_sorted['Value'],
                    y=reason_df_sorted['Reason'],
                    orientation='h',
                    marker_color=MAPP_PURPLE,
                    text=reason_df_sorted['Value'].apply(lambda x: format_currency(x)),
                    textposition='outside'
                ))

                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=350,
                    margin=dict(l=120, r=80, t=20, b=40),
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Return Value (£)'),
                    yaxis=dict(showgrid=False),
                    title='Financial Impact by Reason'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Reasons by category heatmap
            st.markdown('<div class="section-header">Return Reasons by Category</div>', unsafe_allow_html=True)

            reason_by_cat = return_reasons['reason_by_category']
            if len(reason_by_cat) > 0:
                fig = px.imshow(
                    reason_by_cat.values,
                    x=reason_by_cat.columns.tolist(),
                    y=reason_by_cat.index.tolist(),
                    color_continuous_scale=[[0, 'white'], [0.5, MAPP_LIGHT_PURPLE], [1, MAPP_PURPLE]],
                    aspect='auto',
                    text_auto='.0%'
                )

                fig.update_layout(
                    height=400,
                    margin=dict(l=100, r=40, t=20, b=80),
                    xaxis=dict(title='Return Reason', tickangle=-45),
                    yaxis=dict(title='Category'),
                    coloraxis_colorbar=dict(title='%', tickformat='.0%')
                )

                st.plotly_chart(fig, use_container_width=True)

            # Size-related reasons analysis
            st.markdown('<div class="section-header">Size-Related Returns Deep Dive</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                size_reason_data = return_reasons['size_reason_by_size']
                if len(size_reason_data) > 0 and 'pct_too_small' in size_reason_data.columns:
                    # Order sizes properly
                    size_order = ['4', '6', '8', '10', '12', '14', '16', '18', 'XS', 'S', 'M', 'L', 'XL']
                    size_reason_data = size_reason_data.reindex([s for s in size_order if s in size_reason_data.index])

                    fig = go.Figure()

                    if 'Too small' in size_reason_data.columns:
                        fig.add_trace(go.Bar(
                            name='Too small',
                            x=size_reason_data.index,
                            y=size_reason_data['Too small'],
                            marker_color=MAPP_PINK
                        ))

                    if 'Too large' in size_reason_data.columns:
                        fig.add_trace(go.Bar(
                            name='Too large',
                            x=size_reason_data.index,
                            y=size_reason_data['Too large'],
                            marker_color=MAPP_PURPLE
                        ))

                    fig.update_layout(
                        barmode='stack',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300,
                        margin=dict(l=40, r=40, t=30, b=40),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        xaxis=dict(showgrid=False, title='Size'),
                        yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Number of Returns'),
                        title='Size Returns: Too Small vs Too Large'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No size-related return data available")

            with col2:
                # Fit type analysis
                sizing_by_fit = return_reasons['sizing_by_fit']
                if len(sizing_by_fit) > 0:
                    fit_df = sizing_by_fit.reset_index()
                    fit_df_melted = fit_df.melt(id_vars='fit', var_name='Reason', value_name='Count')

                    fig = px.bar(
                        fit_df_melted,
                        x='fit',
                        y='Count',
                        color='Reason',
                        barmode='group',
                        color_discrete_map={'Too small': MAPP_PINK, 'Too large': MAPP_PURPLE}
                    )

                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300,
                        margin=dict(l=40, r=40, t=30, b=40),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        xaxis=dict(showgrid=False, title='Fit Type'),
                        yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Number of Returns'),
                        title='Size Issues by Fit Type'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fit-related sizing data available")

            # Reasons by customer segment
            st.markdown('<div class="section-header">Return Reasons by Customer Segment</div>', unsafe_allow_html=True)

            reason_by_seg = return_reasons['reason_by_segment']
            if len(reason_by_seg) > 0:
                seg_df = reason_by_seg.reset_index()
                seg_melted = seg_df.melt(id_vars='customer_segment', var_name='Reason', value_name='Percentage')

                fig = px.bar(
                    seg_melted,
                    x='customer_segment',
                    y='Percentage',
                    color='Reason',
                    barmode='stack',
                    color_discrete_sequence=[MAPP_PURPLE, MAPP_PINK, MAPP_LIGHT_PURPLE,
                                            '#F472B6', '#A78BFA', '#6366F1', '#818CF8']
                )

                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(l=40, r=40, t=20, b=40),
                    yaxis_tickformat='.0%',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    xaxis=dict(showgrid=False, title='Customer Segment'),
                    yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Percentage of Returns'),
                )

                st.plotly_chart(fig, use_container_width=True)

            # Key insights callout
            changed_mind_pct = overall_stats.loc['Changed mind', 'percentage'] if 'Changed mind' in overall_stats.index else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {MAPP_PURPLE}, {MAPP_LIGHT_PURPLE});
                        color: white; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
                <div style="font-size: 1.1rem; opacity: 0.9;">💡 Key Return Reasons Insights</div>
                <div style="margin-top: 1rem;">
                    <p><strong>Sizing ({sizing_pct:.0%}):</strong> The largest driver of returns. Improve size guides,
                       add garment measurements, and consider fit-specific guidance for slim/fitted items.</p>
                    <p><strong>Quality ({quality_pct:.0%}):</strong> Review QC processes, especially for trend/clearance items
                       which show higher quality-related returns.</p>
                    <p><strong>Changed Mind ({changed_mind_pct:.0%}):</strong>
                       Higher among serial returners - improve product imagery and descriptions to set better expectations.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Return reasons data is not available. Please regenerate the synthetic data with return reasons enabled.")

    # ==========================================================================
    # TAB: HVHR PRODUCTS
    # ==========================================================================
    with tab_hvhr:
        st.markdown('<div class="section-header">High Volume, High Return Products</div>', unsafe_allow_html=True)

        hvhr = analysis['hvhr_products']

        if len(hvhr) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_metric_card(
                    "HVHR Products Identified",
                    len(hvhr),
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "Total Excess Returns",
                    format_number(hvhr['excess_returns'].sum()),
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Est. Cost Impact",
                    format_currency(hvhr['cost_impact'].sum()),
                ), unsafe_allow_html=True)
            
            # HVHR scatter plot
            st.markdown("#### Return Rate vs Volume")
            
            fig = px.scatter(
                hvhr,
                x='qty_purchased',
                y='return_rate',
                size='hvhr_score',
                color='category',
                hover_data=['product_id', 'product_name', 'excess_return_rate'],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis_tickformat='.0%',
                height=400,
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Items Sold'),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Return Rate'),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # HVHR table
            st.markdown("#### Top HVHR Products")
            
            hvhr_display = hvhr[['product_id', 'product_name', 'category', 'qty_purchased', 
                                'return_rate', 'category_baseline', 'excess_return_rate', 
                                'hvhr_score']].head(20).copy()
            
            hvhr_display.columns = ['Product ID', 'Product Name', 'Category', 'Qty Sold',
                                   'Return Rate', 'Category Baseline', 'Excess Rate', 'HVHR Score']
            
            hvhr_display['Return Rate'] = hvhr_display['Return Rate'].apply(lambda x: f"{x:.1%}")
            hvhr_display['Category Baseline'] = hvhr_display['Category Baseline'].apply(lambda x: f"{x:.1%}")
            hvhr_display['Excess Rate'] = hvhr_display['Excess Rate'].apply(lambda x: f"{x:+.1%}")
            hvhr_display['HVHR Score'] = hvhr_display['HVHR Score'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(hvhr_display, use_container_width=True, hide_index=True)
        else:
            st.info("No HVHR products identified with current settings.")
    
    # ==========================================================================
    # TAB: ATTRIBUTE DRIVERS
    # ==========================================================================
    with tab_drivers:
        st.markdown('<div class="section-header">Attribute Drivers of Returns</div>', unsafe_allow_html=True)
        
        attr_drivers = analysis['attribute_drivers']
        
        if len(attr_drivers) > 0:
            # Create grid of charts
            attributes_to_show = ['fit', 'pattern', 'sleeve_style', 'neckline']
            
            for i in range(0, len(attributes_to_show), 2):
                cols = st.columns(2)
                
                for j, attr in enumerate(attributes_to_show[i:i+2]):
                    if attr in attr_drivers:
                        with cols[j]:
                            st.markdown(f"#### {attr.replace('_', ' ').title()}")
                            
                            df = attr_drivers[attr].reset_index()
                            df = df.sort_values('lift', ascending=True).tail(10)
                            
                            colors = [MAPP_PINK if l > 0 else MAPP_PURPLE for l in df['lift']]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=df['lift'],
                                y=df[attr],
                                orientation='h',
                                marker_color=colors,
                                text=df['lift'].apply(lambda x: f"{x:+.1%}"),
                                textposition='outside'
                            ))
                            
                            fig.add_vline(x=0, line_color=MAPP_GRAY, line_width=1)
                            
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                xaxis_tickformat='.0%',
                                height=300,
                                margin=dict(l=100, r=60, t=20, b=40),
                                showlegend=False,
                                xaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Lift vs Baseline'),
                                yaxis=dict(showgrid=False),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; 
                        border: 1px solid #E5E7EB;">
                <h4 style="margin-top: 0; color: {MAPP_PURPLE};">🔍 Key Attribute Findings</h4>
                <p><strong>Positive lift</strong> (pink bars) = higher return rate than average</p>
                <p><strong>Negative lift</strong> (purple bars) = lower return rate than average</p>
                <p>Focus PDP improvements and imagery reviews on attributes with highest positive lift.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No attribute driver data available.")
    
    # ==========================================================================
    # TAB: CUSTOMERS
    # ==========================================================================
    with tab_customers:
        st.markdown('<div class="section-header">Customer Segmentation</div>', unsafe_allow_html=True)
        
        cust_analysis = analysis['customer_analysis']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Total Customers",
                format_number(cust_analysis['total_customers']),
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                "Never Return",
                f"{cust_analysis['pct_never_return']:.0%}",
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Top 10% Drive",
                f"{cust_analysis['top_10_pct_returns']:.0%} of returns",
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "Unprofitable",
                format_number(cust_analysis['unprofitable_customers']),
            ), unsafe_allow_html=True)
        
        # Segment breakdown
        st.markdown("#### Customer Segments")
        
        seg_stats = cust_analysis['segment_stats'].reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                seg_stats,
                values='num_customers',
                names='customer_segment',
                color_discrete_sequence=[MAPP_PURPLE, MAPP_LIGHT_PURPLE, MAPP_PINK, '#F472B6'],
                hole=0.4
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                title='Customers by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                seg_stats,
                values='qty_returned',
                names='customer_segment',
                color_discrete_sequence=[MAPP_PURPLE, MAPP_LIGHT_PURPLE, MAPP_PINK, '#F472B6'],
                hole=0.4
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                title='Returns by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment table
        seg_display = seg_stats.copy()
        seg_display.columns = ['Segment', 'Customers', 'Items Purchased', 'Items Returned', 
                              'Total Profit', 'Return Rate', '% of All Returns']
        seg_display['Return Rate'] = seg_display['Return Rate'].apply(lambda x: f"{x:.1%}")
        seg_display['% of All Returns'] = seg_display['% of All Returns'].apply(lambda x: f"{x:.1%}")
        seg_display['Total Profit'] = seg_display['Total Profit'].apply(lambda x: format_currency(x))
        
        st.dataframe(seg_display, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # TAB: ACTIONS
    # ==========================================================================
    with tab_actions:
        st.markdown('<div class="section-header">Recommended Actions</div>', unsafe_allow_html=True)
        
        recommendations = analysis['recommendations']
        
        if len(recommendations) > 0:
            # Group by priority
            high_priority = [r for r in recommendations if r['priority'] == 'High']
            medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
            low_priority = [r for r in recommendations if r['priority'] == 'Low']
            
            if high_priority:
                st.markdown("#### 🔴 High Priority")
                for rec in high_priority:
                    st.markdown(create_recommendation_card(rec), unsafe_allow_html=True)
            
            if medium_priority:
                st.markdown("#### 🟡 Medium Priority")
                for rec in medium_priority:
                    st.markdown(create_recommendation_card(rec), unsafe_allow_html=True)
            
            if low_priority:
                st.markdown("#### 🟢 Low Priority")
                for rec in low_priority:
                    st.markdown(create_recommendation_card(rec), unsafe_allow_html=True)
            
            # Export button
            st.markdown("---")
            
            rec_df = pd.DataFrame(recommendations)
            csv = rec_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Export Recommendations as CSV",
                data=csv,
                file_name="mapp_returns_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.success("No critical issues identified - returns are within acceptable ranges!")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {MAPP_GRAY}; padding: 1rem 0;">
        <span style="font-weight: 600;">Mapp</span><span style="color: {MAPP_PINK}; font-weight: 600;">Fashion</span> 
        Returns Intelligence Agent • Powered by AI
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
