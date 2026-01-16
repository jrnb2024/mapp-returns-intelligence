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
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from analysis_engine import ReturnsAnalyzer
from decision_engine import DecisionEngine
from ledger import DecisionLedger
from simulator import ActionSimulator
from impact_calculator import ImpactCalculator
from actions import Action, ActionType, ActionStatus

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

    /* Agent-specific styles */
    .decision-card {{
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .decision-card-suppress {{
        border-left: 4px solid #EF4444;
    }}

    .decision-card-warn {{
        border-left: 4px solid #F59E0B;
    }}

    .decision-card-resegment {{
        border-left: 4px solid #3B82F6;
    }}

    .decision-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}

    .decision-type {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }}

    .type-suppress {{
        background: #FEE2E2;
        color: #DC2626;
    }}

    .type-warn {{
        background: #FEF3C7;
        color: #D97706;
    }}

    .type-resegment {{
        background: #DBEAFE;
        color: #2563EB;
    }}

    .confidence-bar {{
        height: 8px;
        border-radius: 4px;
        background: #E5E7EB;
        overflow: hidden;
        margin: 0.5rem 0;
    }}

    .confidence-fill {{
        height: 100%;
        border-radius: 4px;
    }}

    .confidence-high {{
        background: #10B981;
    }}

    .confidence-medium {{
        background: #F59E0B;
    }}

    .confidence-low {{
        background: #EF4444;
    }}

    .thinking-step {{
        background: #F9FAFB;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid {MAPP_PURPLE};
    }}

    .thinking-step-title {{
        font-weight: 600;
        color: {MAPP_PURPLE};
        margin-bottom: 0.5rem;
    }}

    .status-pending {{
        color: #6B7280;
    }}

    .status-approved {{
        color: #10B981;
    }}

    .status-rejected {{
        color: #EF4444;
    }}

    .status-executed {{
        color: #3B82F6;
    }}

    .api-preview {{
        background: #1F2937;
        color: #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }}

    .integration-card {{
        background: linear-gradient(135deg, #1F2937, #374151);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
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
# AGENT TAB FUNCTIONS
# =============================================================================

def get_action_icon(action_type: str) -> str:
    """Get icon for action type."""
    icons = {
        'SUPPRESS': '🚫',
        'WARN': '⚠️',
        'RESEGMENT': '🔄',
    }
    return icons.get(action_type, '📋')

def get_status_icon(status: str) -> str:
    """Get icon for status."""
    icons = {
        'PENDING': '⏳',
        'APPROVED': '✅',
        'REJECTED': '❌',
        'EXECUTED': '🚀',
    }
    return icons.get(status, '❓')

def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 0.8:
        return 'high'
    elif confidence >= 0.6:
        return 'medium'
    return 'low'

def render_agent_tab(transactions_df, analysis):
    """Render the Agent tab with all sections."""

    # Initialize ledger
    ledger = DecisionLedger()

    # =========================================================================
    # SECTION 0: Demo Controls
    # =========================================================================
    st.markdown('<div class="section-header">Agent Controls</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🤖 Run Agent Analysis", type="primary", use_container_width=True):
            with st.spinner("Agent analyzing returns data..."):
                time.sleep(1.5)  # Demo effect

                analyzer = ReturnsAnalyzer(transactions_df)
                engine = DecisionEngine(analyzer)
                decisions = engine.generate_decisions()

                ledger.clear()
                ledger.record_decisions(decisions)

                st.success(f"✅ Agent generated {len(decisions)} decisions")
                time.sleep(0.5)
                st.rerun()

    with col2:
        if st.button("🔄 Reset Demo", use_container_width=True):
            ledger.clear()
            st.success("Demo reset - ready for fresh run")
            time.sleep(0.5)
            st.rerun()

    with col3:
        approved = ledger.get_decisions(status=ActionStatus.APPROVED)
        if len(approved) > 0:
            if st.button("▶️ Simulate Execution", use_container_width=True):
                st.session_state['show_simulation'] = True

    # =========================================================================
    # SECTION 1: Agent Dashboard
    # =========================================================================
    st.markdown('<div class="section-header">Agent Dashboard</div>', unsafe_allow_html=True)

    stats = ledger.get_summary_stats()

    # Check if we have decisions
    if stats['total_decisions'] == 0:
        st.info("👆 Click **Run Agent Analysis** to generate decisions from the returns data.")

        # Show what the agent will do
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; border: 1px solid #E5E7EB;">
            <h4 style="margin-top: 0;">What the Agent Does</h4>
            <p>The Returns Intelligence Agent analyzes your transaction data and generates three types of actions:</p>
            <ul>
                <li><strong>🚫 SUPPRESS</strong> - Remove high-return products from recommendations (Dressipi API)</li>
                <li><strong>⚠️ WARN</strong> - Add sizing warnings to product pages (Mapp Engage)</li>
                <li><strong>🔄 RESEGMENT</strong> - Move customers to return-risk segments (Mapp Intelligence)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get impact calculation
    all_decisions = ledger.get_decisions()
    analyzer = ReturnsAnalyzer(transactions_df)
    calculator = ImpactCalculator(analyzer)
    portfolio_impact = calculator.calculate_portfolio_impact(all_decisions)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_metric_card(
            "Decisions Generated",
            f"{stats['total_decisions']}",
        ), unsafe_allow_html=True)

    with col2:
        pending_count = stats['by_status'].get('PENDING', 0)
        st.markdown(create_metric_card(
            "Pending Review",
            f"{pending_count}",
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card(
            "Est. Return Reduction",
            f"{portfolio_impact['return_rate_reduction_pct']}",
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card(
            "Est. Cost Savings",
            f"£{portfolio_impact['total_return_cost_saved']:,.0f}",
        ), unsafe_allow_html=True)

    # Breakdown by type
    st.markdown("#### Decisions by Type")

    type_cols = st.columns(3)

    type_info = [
        ('SUPPRESS', '🚫', '#EF4444', 'Products to remove from recommendations'),
        ('WARN', '⚠️', '#F59E0B', 'Products needing sizing warnings'),
        ('RESEGMENT', '🔄', '#3B82F6', 'Customers to move to new segments'),
    ]

    for i, (type_name, icon, color, desc) in enumerate(type_info):
        count = stats['by_type'].get(type_name, 0)
        with type_cols[i]:
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1rem; border-left: 4px solid {color};">
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{icon} {count}</div>
                <div style="font-size: 0.85rem; color: #6B7280;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Show simulation results if triggered
    if st.session_state.get('show_simulation', False):
        st.markdown("---")
        st.markdown("#### 📡 Execution Simulation")

        approved = ledger.get_decisions(status=ActionStatus.APPROVED)
        simulator = ActionSimulator()
        batch_sim = simulator.generate_batch_simulation(approved)

        st.markdown(f"""
        <div style="background: #F0FDF4; border-radius: 12px; padding: 1rem; border: 1px solid #86EFAC;">
            <strong>Simulation Complete</strong><br>
            {len(approved)} decisions would be executed via {batch_sim['api_calls_required']} API calls<br>
            Estimated execution time: {batch_sim['estimated_execution_time_seconds']:.1f} seconds
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Execution Details"):
            st.json(batch_sim)

        st.session_state['show_simulation'] = False

    # =========================================================================
    # SECTION 2: Decision Queue
    # =========================================================================
    st.markdown('<div class="section-header">Decision Queue</div>', unsafe_allow_html=True)

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All", "SUPPRESS", "WARN", "RESEGMENT"],
            key="type_filter"
        )

    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Confidence (High to Low)", "Impact (High to Low)", "Newest First"],
            key="sort_by"
        )

    with filter_col3:
        # Bulk actions
        st.markdown("<div style='padding-top: 1.7rem;'>", unsafe_allow_html=True)
        bulk_col1, bulk_col2 = st.columns(2)
        with bulk_col1:
            if st.button("✅ Approve High Confidence (≥80%)", use_container_width=True):
                count = ledger.approve_decisions(min_confidence=0.8)
                if count > 0:
                    st.success(f"Approved {count} high-confidence decisions")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.info("No pending high-confidence decisions to approve")
        st.markdown("</div>", unsafe_allow_html=True)

    # Get filtered decisions
    action_type = None if type_filter == "All" else ActionType[type_filter]
    decisions = ledger.get_decisions(action_type=action_type, status=ActionStatus.PENDING)

    # Sort decisions
    if sort_by == "Confidence (High to Low)":
        decisions.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "Impact (High to Low)":
        decisions.sort(key=lambda x: x.estimated_impact.get('net_margin_impact', 0), reverse=True)

    # Display decisions (paginated)
    if len(decisions) == 0:
        st.info("No pending decisions matching filters. Try changing filters or run the agent again.")
    else:
        # Pagination
        page_size = 10
        total_pages = (len(decisions) - 1) // page_size + 1

        if 'queue_page' not in st.session_state:
            st.session_state.queue_page = 0

        start_idx = st.session_state.queue_page * page_size
        end_idx = min(start_idx + page_size, len(decisions))

        st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(decisions)} pending decisions")

        for decision in decisions[start_idx:end_idx]:
            render_decision_card(decision, ledger, analyzer)

        # Pagination controls
        if total_pages > 1:
            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            with page_col1:
                if st.button("← Previous", disabled=st.session_state.queue_page == 0):
                    st.session_state.queue_page -= 1
                    st.rerun()
            with page_col2:
                st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>Page {st.session_state.queue_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with page_col3:
                if st.button("Next →", disabled=st.session_state.queue_page >= total_pages - 1):
                    st.session_state.queue_page += 1
                    st.rerun()

    # =========================================================================
    # SECTION 3: Agent Thinking
    # =========================================================================
    st.markdown('<div class="section-header">How the Agent Thinks</div>', unsafe_allow_html=True)

    # Decision selector
    all_decisions = ledger.get_decisions(limit=50)
    if len(all_decisions) > 0:
        decision_options = {
            f"{get_action_icon(d.action_type.value)} {d.action_type.value}: {d.target} ({d.confidence:.0%})": d
            for d in all_decisions[:20]
        }

        selected_label = st.selectbox(
            "Select a decision to inspect",
            list(decision_options.keys()),
            key="thinking_selector"
        )

        selected_decision = decision_options[selected_label]
        render_agent_thinking(selected_decision, analyzer)
    else:
        st.info("Run the agent to see decision reasoning.")

    # =========================================================================
    # SECTION 4: Decision History
    # =========================================================================
    st.markdown('<div class="section-header">Decision History</div>', unsafe_allow_html=True)

    # Status filter
    history_col1, history_col2 = st.columns([1, 3])

    with history_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "PENDING", "APPROVED", "REJECTED", "EXECUTED"],
            key="history_status"
        )

    # Get filtered history
    status = None if status_filter == "All" else ActionStatus[status_filter]
    history_decisions = ledger.get_decisions(status=status)

    if len(history_decisions) > 0:
        # Create DataFrame for display
        history_data = []
        for d in history_decisions[:100]:  # Limit to 100 for performance
            history_data.append({
                'Timestamp': d.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Type': f"{get_action_icon(d.action_type.value)} {d.action_type.value}",
                'Target': d.target,
                'Confidence': f"{d.confidence:.0%}",
                'Status': f"{get_status_icon(d.status.value)} {d.status.value}",
                'Impact': f"£{d.estimated_impact.get('net_margin_impact', 0):,.2f}",
            })

        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Cumulative impact of approved/executed
        cumulative = ledger.get_cumulative_impact()
        if cumulative['total_decisions_counted'] > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {MAPP_PURPLE}, {MAPP_LIGHT_PURPLE});
                        color: white; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                <strong>Cumulative Impact of Approved Decisions:</strong>
                £{cumulative['total_net_margin_impact']:,.2f} estimated margin impact
                from {cumulative['total_decisions_counted']} decisions
            </div>
            """, unsafe_allow_html=True)

        # Export button
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Decision History",
            data=csv,
            file_name="agent_decision_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No decisions matching filter.")

    # =========================================================================
    # SECTION 5: Integration Points
    # =========================================================================
    st.markdown('<div class="section-header">Production Integration</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="integration-card">
        <h4 style="margin-top: 0; color: white;">📡 Production Integration Points</h4>
        <p style="opacity: 0.9;">This demo simulates API calls. In production, these would connect to:</p>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem;">
                <strong>🚫 Dressipi API (SUPPRESS)</strong><br>
                <code style="font-size: 0.8rem;">/v1/facetted/search</code><br>
                <span style="font-size: 0.85rem; opacity: 0.8;">Auth: JWT via x-dressipi-jwt header</span>
            </div>

            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem;">
                <strong>⚠️ Mapp Engage (WARN)</strong><br>
                <code style="font-size: 0.8rem;">/api/v2/triggers</code><br>
                <span style="font-size: 0.85rem; opacity: 0.8;">Auth: API key + system user</span>
            </div>

            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem;">
                <strong>🔄 Mapp Intelligence (RESEGMENT)</strong><br>
                <code style="font-size: 0.8rem;">/api/v1/segments/members</code><br>
                <span style="font-size: 0.85rem; opacity: 0.8;">Auth: OAuth 2.0</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_decision_card(decision: Action, ledger: DecisionLedger, analyzer: ReturnsAnalyzer):
    """Render a single decision card with approve/reject buttons."""

    action_type = decision.action_type.value.lower()
    confidence_class = get_confidence_class(decision.confidence)
    icon = get_action_icon(decision.action_type.value)

    # Card container
    with st.container():
        st.markdown(f"""
        <div class="decision-card decision-card-{action_type}">
            <div class="decision-header">
                <span class="decision-type type-{action_type}">{icon} {decision.action_type.value}</span>
                <span style="font-weight: 600;">{decision.target}</span>
            </div>
            <div style="font-size: 0.95rem; color: #374151; margin-bottom: 0.75rem;">
                {decision.reason}
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill confidence-{confidence_class}" style="width: {decision.confidence * 100}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #6B7280;">
                <span>Confidence: {decision.confidence:.0%}</span>
                <span>Impact: £{decision.estimated_impact.get('net_margin_impact', 0):,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

        with btn_col1:
            if st.button("✅ Approve", key=f"approve_{decision.id}", use_container_width=True):
                ledger.update_status(decision.id, ActionStatus.APPROVED, "Approved by user")
                st.success(f"Approved: {decision.target}")
                time.sleep(0.3)
                st.rerun()

        with btn_col2:
            if st.button("❌ Reject", key=f"reject_{decision.id}", use_container_width=True):
                ledger.update_status(decision.id, ActionStatus.REJECTED, "Rejected by user")
                st.warning(f"Rejected: {decision.target}")
                time.sleep(0.3)
                st.rerun()


def render_agent_thinking(decision: Action, analyzer: ReturnsAnalyzer):
    """Render the 'Agent Thinking' explanation for a decision."""

    with st.expander(f"🧠 Agent Reasoning: {decision.target}", expanded=True):
        data = decision.supporting_data

        # Step 1: Data Collection
        st.markdown(f"""
        <div class="thinking-step">
            <div class="thinking-step-title">Step 1: Data Collection</div>
            <code>
            ├── Target: {decision.target}<br>
            ├── Type: {decision.action_type.value}<br>
            """, unsafe_allow_html=True)

        if decision.action_type == ActionType.SUPPRESS:
            st.markdown(f"""
            ├── Product: {data.get('product_name', 'N/A')}<br>
            ├── Category: {data.get('category', 'N/A')}<br>
            ├── Volume: {data.get('volume', 0):,} units sold<br>
            └── Returns: {data.get('returns_count', 0):,} units returned
            </code>
        </div>
            """, unsafe_allow_html=True)
        elif decision.action_type == ActionType.WARN:
            st.markdown(f"""
            ├── Product: {data.get('product_name', 'N/A')}<br>
            ├── Category: {data.get('category', 'N/A')}<br>
            ├── Sizing Issue Rate: {data.get('combined_sizing_issue', 0):.0%}<br>
            └── Dominant Issue: {data.get('skew_direction', 'unknown')}
            </code>
        </div>
            """, unsafe_allow_html=True)
        else:  # RESEGMENT
            st.markdown(f"""
            ├── Customer: {data.get('customer_id', 'N/A')}<br>
            ├── Current Segment: {data.get('current_segment', 'N/A')}<br>
            ├── Recent Returns: {data.get('recent_returns', 0)}<br>
            └── Recent Return Rate: {data.get('recent_return_rate', 0):.0%}
            </code>
        </div>
            """, unsafe_allow_html=True)

        # Step 2: Analysis
        st.markdown(f"""
        <div class="thinking-step">
            <div class="thinking-step-title">Step 2: Analysis</div>
            <code>
        """, unsafe_allow_html=True)

        if decision.action_type == ActionType.SUPPRESS:
            st.markdown(f"""
            ├── Return Rate: {data.get('return_rate', 0):.1%}<br>
            ├── Category Baseline: {data.get('category_baseline', 0):.1%}<br>
            ├── Multiplier vs Category: {data.get('multiplier_vs_category', 0):.1f}x<br>
            └── Excess Return Rate: {data.get('excess_return_rate', 0):+.1%}
            </code>
        </div>
            """, unsafe_allow_html=True)
        elif decision.action_type == ActionType.WARN:
            st.markdown(f"""
            ├── Return Rate: {data.get('return_rate', 0):.1%}<br>
            ├── Sizing Issue Rate: {data.get('sizing_issue_rate', 0):.1%}<br>
            ├── Size Skew Score: {data.get('size_skew_score', 0):.2f}<br>
            └── Direction: Runs {data.get('skew_direction', 'unknown')}
            </code>
        </div>
            """, unsafe_allow_html=True)
        else:  # RESEGMENT
            st.markdown(f"""
            ├── Recent Returns: {data.get('recent_returns', 0)} in {data.get('period_days', 90)} days<br>
            ├── Return Rate: {data.get('recent_return_rate', 0):.1%}<br>
            ├── Threshold: 5 returns or 50% rate<br>
            └── Suggested Segment: {data.get('suggested_segment', 'N/A')}
            </code>
        </div>
            """, unsafe_allow_html=True)

        # Step 3: Decision
        st.markdown(f"""
        <div class="thinking-step">
            <div class="thinking-step-title">Step 3: Decision</div>
            <code>
            ├── Action: {decision.action_type.value}<br>
            ├── Confidence: {decision.confidence:.0%}<br>
            └── Status: {decision.status.value}
            </code>
        </div>
        """, unsafe_allow_html=True)

        # Step 4: Impact Estimate
        impact = decision.estimated_impact
        st.markdown(f"""
        <div class="thinking-step">
            <div class="thinking-step-title">Step 4: Impact Estimate</div>
            <code>
            ├── Return Reduction: {impact.get('return_reduction', 0):.2%} of total returns<br>
            ├── CVR Impact: {impact.get('cvr_impact', 0):+.1%}<br>
            └── Net Margin Impact: £{impact.get('net_margin_impact', 0):,.2f}
            </code>
        </div>
        """, unsafe_allow_html=True)

        # Step 5: Execution Preview
        api_call = decision.mapp_api_call
        st.markdown(f"""
        <div class="thinking-step">
            <div class="thinking-step-title">Step 5: Execution (Simulated)</div>
            <code>
            ├── Endpoint: {api_call.get('method', 'POST')} {api_call.get('endpoint', 'N/A')}<br>
            ├── Payload: {str(api_call.get('params', {}))[:80]}...<br>
            └── Rollback: Reverse the action via same endpoint
            </code>
        </div>
        """, unsafe_allow_html=True)


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
    tab_overview, tab_category, tab_size, tab_hvhr, tab_drivers, tab_customers, tab_actions, tab_agent = st.tabs([
        "📈 Overview",
        "🏷️ Categories",
        "📏 Sizing",
        "⚠️ HVHR Products",
        "🔍 Attribute Drivers",
        "👥 Customers",
        "✅ Actions",
        "🤖 Agent"
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

    # ==========================================================================
    # TAB: AGENT
    # ==========================================================================
    with tab_agent:
        render_agent_tab(transactions, analysis)

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
