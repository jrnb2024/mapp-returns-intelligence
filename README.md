# Mapp Fashion Returns Intelligence Agent

> AI-powered returns analysis for fashion retailers. Identifies high-return products, analyzes drivers, and generates actionable recommendations.

![Mapp Fashion](docs/screenshot.png)

## 🎯 What This Does

This agent analyzes your fashion retail transaction data to:

1. **Identify HVHR Products** - High Volume, High Return items that need attention
2. **Analyze Return Drivers** - Which attributes (fit, pattern, size) drive returns
3. **Segment Customers** - Understand return behavior by customer type
4. **Benchmark Categories** - Compare your return rates to industry standards
5. **Generate Recommendations** - Actionable insights for merchandisers

## 📊 Analysis Modules

| Module | What It Does |
|--------|--------------|
| **Executive Summary** | Overall return rate, financial impact, trend |
| **Category Benchmark** | Return rate by category vs industry baseline |
| **Size Analysis** | Return patterns by size (smaller = higher returns?) |
| **HVHR Detection** | Products with excess return rates + high volume |
| **Attribute Drivers** | Lift analysis for fit, pattern, neckline, etc. |
| **Customer Segments** | Serial returners vs profitable customers |
| **Multi-Buy Analysis** | Wardrobing and multi-size purchase patterns |
| **Time Trends** | Monthly return rate tracking |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (Demo)

```bash
python src/generate_synthetic_data.py
```

This creates realistic transaction data based on patterns from actual fashion returns analysis:
- 50,000 transactions
- 1,000 products across 11 categories
- 5,000 customers in 4 segments
- Full year of data (2024)

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
mapp-returns-agent/
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── data/
│   ├── transactions.csv      # Generated transaction data
│   ├── products.csv          # Product catalog
│   └── customers.csv         # Customer segments
└── src/
    ├── generate_synthetic_data.py  # Data generator
    └── analysis_engine.py          # Core analysis logic
```

## 📈 Using Your Own Data

To use real data, prepare CSV files with these columns:

### transactions.csv (required)
| Column | Type | Description |
|--------|------|-------------|
| order_id | string | Unique order identifier |
| customer_id | string | Customer identifier |
| product_id | string | Product identifier |
| category | string | Product category |
| size | string | Size purchased |
| order_date | date | Order date (YYYY-MM-DD) |
| qty_purchased | int | Quantity purchased (usually 1) |
| qty_returned | int | Quantity returned (0 or 1) |
| value_purchased | float | Purchase value |
| value_returned | float | Return value |

### Optional: Product attributes
Add these columns to transactions.csv for attribute analysis:
- `fit` (Slim, Regular, Relaxed, etc.)
- `pattern` (Plain, Floral, Stripe, etc.)
- `neckline` (V-Neck, Round, Crew, etc.)
- `sleeve_style` (Standard, Balloon, Puff, etc.)

### Optional: Customer segment tracking
- `customer_segment` (for pre-defined segments)
- `is_multi_style_order` (boolean)
- `is_multi_size_order` (boolean)

## 💡 Key Insights Generated

### Financial Impact
> "Every 1% reduction in returns = £21,412 EBIT"

### HVHR Products
Identifies products like:
> "AW24 Trend Fitted Animal Print Dress - 52% return rate (+14% vs category)"

### Attribute Drivers
Shows which attributes increase returns:
> "Balloon sleeves: +4% lift vs baseline"
> "Size 6: +10% lift vs baseline"

### Customer Segments
> "Top 10% of customers drive 63% of returns"

## 🔧 Configuration

Key settings in `src/analysis_engine.py`:

```python
# Financial assumptions
COST_PER_RETURN = 1.08        # £ per returned item
GROSS_MARGIN = 0.685          # 68.5%

# HVHR thresholds
min_qty = 30                  # Minimum volume for HVHR
min_sample_size = 50          # Minimum for attribute analysis
```

## 📱 Dashboard Features

- **Overview Tab**: Executive summary with KPIs and trends
- **Categories Tab**: Benchmark against industry standards
- **Sizing Tab**: Identify size-specific issues
- **HVHR Products Tab**: Drill into problematic products
- **Attribute Drivers Tab**: Visualize return drivers
- **Customers Tab**: Segment analysis
- **Actions Tab**: Prioritized recommendations with export

## 🎨 Mapp Branding

The dashboard uses Mapp's brand colors:
- Primary Purple: `#5B21B6`
- Accent Pink: `#EC4899`

## 📋 Based On

Analysis methodology inspired by the [Hush Returns Consultancy](https://dressipi.com) which demonstrated:
- 38% baseline return rate reduced through targeted actions
- £532K value per 1% return rate reduction
- Size-specific issues: small sizes return 10% more
- Customer concentration: 10% of customers = 63% of returns

## 🤝 Support

For questions about Mapp Fashion or the Returns Intelligence Agent:
- Contact: [Your Mapp representative]
- Documentation: [Mapp Fashion docs]

---

*Built with ❤️ by Mapp Fashion AI*
