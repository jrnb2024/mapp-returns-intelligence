"""
Synthetic Data Generator for Mapp Fashion Returns Intelligence Agent

Generates realistic transaction data with patterns matching Sarah's Hush analysis:
- Customer segments with different return behaviors
- Size-based return patterns (small sizes return 10% more)
- Category-based return rates (Dresses highest, Tops lowest)
- Multi-buy patterns (wardrobing)
- Seasonal trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION - Based on Sarah's Hush Report Findings
# =============================================================================

NUM_TRANSACTIONS = 50000
NUM_CUSTOMERS = 5000
NUM_PRODUCTS = 1000
DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2024, 12, 31)

# Customer segments (from Sarah's report: 10% of customers = 63% of returns)
CUSTOMER_SEGMENTS = {
    'low_returner': {'pct': 0.53, 'return_rate': 0.0, 'description': 'Never returns'},
    'normal_returner': {'pct': 0.35, 'return_rate': 0.35, 'description': 'Returns like average'},
    'high_returner': {'pct': 0.10, 'return_rate': 0.65, 'description': 'Returns most items'},
    'serial_returner': {'pct': 0.02, 'return_rate': 0.92, 'description': 'Returns almost everything'},
}

# Category base return rates (from Sarah's benchmarking)
CATEGORIES = {
    'Dresses': {'base_return_rate': 0.53, 'pct_of_sales': 0.20, 'avg_price': 89},
    'Jeans': {'base_return_rate': 0.55, 'pct_of_sales': 0.08, 'avg_price': 75},
    'Tops': {'base_return_rate': 0.34, 'pct_of_sales': 0.25, 'avg_price': 45},
    'Jumpers': {'base_return_rate': 0.37, 'pct_of_sales': 0.12, 'avg_price': 65},
    'Skirts': {'base_return_rate': 0.45, 'pct_of_sales': 0.06, 'avg_price': 55},
    'Outerwear': {'base_return_rate': 0.53, 'pct_of_sales': 0.05, 'avg_price': 120},
    'Shorts': {'base_return_rate': 0.43, 'pct_of_sales': 0.04, 'avg_price': 40},
    'Jumpsuits': {'base_return_rate': 0.51, 'pct_of_sales': 0.03, 'avg_price': 85},
    'Sweatshirts': {'base_return_rate': 0.34, 'pct_of_sales': 0.08, 'avg_price': 55},
    'Pyjamas': {'base_return_rate': 0.26, 'pct_of_sales': 0.05, 'avg_price': 45},
    'Accessories': {'base_return_rate': 0.15, 'pct_of_sales': 0.04, 'avg_price': 25},
}

# Size return rate modifiers (from Sarah's report: smaller sizes +10%)
SIZE_MODIFIERS = {
    '4': 0.15, '6': 0.10, 'XS': 0.12,
    '8': 0.05, '10': 0.03, 'S': 0.04,
    '12': 0.0, 'M': 0.0,
    '14': -0.02, '16': -0.03, 'L': -0.02,
    '18': -0.04, 'XL': -0.03,
}

# Numeric vs Alpha sizing by category
NUMERIC_SIZED_CATEGORIES = ['Dresses', 'Jeans', 'Skirts', 'Shorts']
ALPHA_SIZED_CATEGORIES = ['Tops', 'Jumpers', 'Outerwear', 'Jumpsuits', 'Sweatshirts', 'Pyjamas']
NON_SIZED_CATEGORIES = ['Accessories']

NUMERIC_SIZES = ['4', '6', '8', '10', '12', '14', '16', '18']
ALPHA_SIZES = ['XS', 'S', 'M', 'L', 'XL']
SIZE_DISTRIBUTION_NUMERIC = [0.02, 0.08, 0.18, 0.25, 0.22, 0.15, 0.07, 0.03]
SIZE_DISTRIBUTION_ALPHA = [0.05, 0.25, 0.35, 0.25, 0.10]

# Product attributes (for attribute driver analysis)
FITS = ['Slim', 'Regular', 'Relaxed', 'Oversized', 'Fitted']
FIT_MODIFIERS = {'Slim': 0.08, 'Regular': 0.0, 'Relaxed': -0.02, 'Oversized': 0.05, 'Fitted': 0.10}

NECKLINES = ['V-Neck', 'Round', 'Crew', 'Boat', 'Square', 'High', 'Off-Shoulder']
NECKLINE_MODIFIERS = {'V-Neck': 0.0, 'Round': -0.02, 'Crew': -0.02, 'Boat': 0.03, 
                      'Square': 0.02, 'High': 0.01, 'Off-Shoulder': 0.08}

PATTERNS = ['Plain', 'Floral', 'Stripe', 'Check', 'Animal Print', 'Abstract', 'Polka Dot']
PATTERN_MODIFIERS = {'Plain': -0.03, 'Floral': 0.02, 'Stripe': 0.0, 'Check': 0.03,
                     'Animal Print': 0.06, 'Abstract': 0.04, 'Polka Dot': 0.01}

SLEEVE_STYLES = ['Standard', 'Sleeveless', 'Balloon', 'Puff', 'Bell', 'Cap', 'Long']
SLEEVE_MODIFIERS = {'Standard': 0.0, 'Sleeveless': 0.02, 'Balloon': 0.04, 
                    'Puff': 0.03, 'Bell': 0.03, 'Cap': 0.01, 'Long': -0.01}

LENGTHS = ['Mini', 'Above Knee', 'Knee', 'Midi', 'Maxi']
LENGTH_MODIFIERS = {'Mini': 0.05, 'Above Knee': 0.02, 'Knee': 0.0, 'Midi': 0.03, 'Maxi': 0.06}

OCCASIONS = ['Casual', 'Smart Casual', 'Work', 'Evening', 'Holiday', 'Loungewear']
STYLE_AESTHETICS = ['Classic', 'Contemporary', 'Boho', 'Minimalist', 'Romantic', 'Sporty']
TRENDS = ['AW24 Core', 'AW24 Trend', 'SS25 Preview', 'Continuity', 'Clearance']
TREND_MODIFIERS = {'AW24 Core': 0.0, 'AW24 Trend': 0.05, 'SS25 Preview': 0.08, 
                   'Continuity': -0.05, 'Clearance': 0.02}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_customers():
    """Generate customer base with segments."""
    customers = []
    customer_id = 1000
    
    for segment, config in CUSTOMER_SEGMENTS.items():
        n_customers = int(NUM_CUSTOMERS * config['pct'])
        for _ in range(n_customers):
            customer_id += 1
            customers.append({
                'customer_id': f'CUST_{customer_id}',
                'segment': segment,
                'base_return_rate': config['return_rate'],
                'description': config['description'],
                # Some randomness per customer
                'return_variance': np.random.uniform(-0.05, 0.05),
            })
    
    return pd.DataFrame(customers)


def generate_products():
    """Generate product catalog with attributes."""
    products = []
    product_counter = 0
    
    for category, config in CATEGORIES.items():
        n_products = int(NUM_PRODUCTS * config['pct_of_sales'])
        
        for i in range(n_products):
            product_counter += 1
            product_id = f'PROD_{product_counter:05d}'
            
            # Assign attributes
            fit = random.choice(FITS)
            neckline = random.choice(NECKLINES) if category in ['Dresses', 'Tops', 'Jumpers', 'Blouses'] else '-'
            pattern = random.choice(PATTERNS)
            sleeve_style = random.choice(SLEEVE_STYLES) if category not in ['Accessories'] else '-'
            length = random.choice(LENGTHS) if category in ['Dresses', 'Skirts'] else '-'
            occasion = random.choice(OCCASIONS)
            style_aesthetic = random.choice(STYLE_AESTHETICS)
            trend = random.choice(TRENDS)
            
            # Calculate product's base return rate
            base_rate = config['base_return_rate']
            base_rate += FIT_MODIFIERS.get(fit, 0)
            base_rate += NECKLINE_MODIFIERS.get(neckline, 0)
            base_rate += PATTERN_MODIFIERS.get(pattern, 0)
            base_rate += SLEEVE_MODIFIERS.get(sleeve_style, 0)
            base_rate += LENGTH_MODIFIERS.get(length, 0)
            base_rate += TREND_MODIFIERS.get(trend, 0)
            
            # Add some product-specific randomness
            base_rate += np.random.uniform(-0.05, 0.05)
            base_rate = np.clip(base_rate, 0.05, 0.95)
            
            # Price with some variance
            price = config['avg_price'] * np.random.uniform(0.7, 1.3)
            
            products.append({
                'product_id': product_id,
                'product_name': f'{trend} {fit} {pattern} {category[:-1] if category.endswith("s") else category}',
                'category': category,
                'fit': fit,
                'neckline': neckline,
                'pattern': pattern,
                'sleeve_style': sleeve_style,
                'length': length,
                'occasion': occasion,
                'style_aesthetic': style_aesthetic,
                'trend': trend,
                'price': round(price, 2),
                'base_return_rate': base_rate,
            })
    
    return pd.DataFrame(products)


def generate_transactions(customers_df, products_df):
    """Generate transactions with realistic patterns."""
    transactions = []
    order_id = 100000
    
    # Pre-compute lookup dicts
    customer_dict = customers_df.set_index('customer_id').to_dict('index')
    product_dict = products_df.set_index('product_id').to_dict('index')
    product_ids = products_df['product_id'].tolist()
    customer_ids = customers_df['customer_id'].tolist()
    
    # Weight products by category sales distribution
    product_weights = []
    for pid in product_ids:
        cat = product_dict[pid]['category']
        product_weights.append(CATEGORIES[cat]['pct_of_sales'])
    product_weights = np.array(product_weights) / sum(product_weights)
    
    # Generate orders
    items_generated = 0
    while items_generated < NUM_TRANSACTIONS:
        order_id += 1
        customer_id = random.choice(customer_ids)
        customer = customer_dict[customer_id]
        
        # Order date with seasonal pattern (more orders in Oct-Dec)
        month_weights = [0.06, 0.06, 0.07, 0.08, 0.08, 0.07, 0.08, 0.09, 0.10, 0.11, 0.10, 0.10]
        month = np.random.choice(range(1, 13), p=month_weights)
        day = random.randint(1, 28)
        order_date = datetime(2024, month, day)
        
        # Determine order type (single item, multi-style, multi-size)
        order_type_roll = random.random()
        if order_type_roll < 0.55:
            # Single item order
            n_items = 1
            is_multi_size = False
        elif order_type_roll < 0.85:
            # Multi-style order (wardrobing)
            n_items = random.randint(2, 5)
            is_multi_size = False
        else:
            # Multi-size order (same item, 2 sizes)
            n_items = 2
            is_multi_size = True
        
        # Select products
        if is_multi_size:
            product_id = np.random.choice(product_ids, p=product_weights)
            selected_products = [product_id, product_id]  # Same product, different sizes
        else:
            selected_products = np.random.choice(product_ids, size=n_items, replace=False, p=product_weights)
        
        for idx, product_id in enumerate(selected_products):
            product = product_dict[product_id]
            category = product['category']
            
            # Determine size
            if category in NUMERIC_SIZED_CATEGORIES:
                size = np.random.choice(NUMERIC_SIZES, p=SIZE_DISTRIBUTION_NUMERIC)
                if is_multi_size and idx == 1:
                    # Second size is adjacent
                    size_idx = NUMERIC_SIZES.index(size)
                    size = NUMERIC_SIZES[min(size_idx + 1, len(NUMERIC_SIZES) - 1)]
            elif category in ALPHA_SIZED_CATEGORIES:
                size = np.random.choice(ALPHA_SIZES, p=SIZE_DISTRIBUTION_ALPHA)
                if is_multi_size and idx == 1:
                    size_idx = ALPHA_SIZES.index(size)
                    size = ALPHA_SIZES[min(size_idx + 1, len(ALPHA_SIZES) - 1)]
            else:
                size = 'One Size'
            
            # Calculate return probability
            return_prob = product['base_return_rate']
            return_prob += SIZE_MODIFIERS.get(size, 0)
            return_prob += customer['return_variance']
            
            # Apply customer segment modifier (multiplicative)
            segment_rate = customer['base_return_rate']
            if segment_rate > 0:
                return_prob = return_prob * (0.5 + segment_rate)  # Blend product and customer rates
            else:
                return_prob = 0  # Low returners don't return
            
            # Multi-buy return patterns
            if n_items > 1 and not is_multi_size:
                # Wardrobing: higher chance of returning some items
                return_prob += 0.10
            if is_multi_size:
                # Multi-size: one usually gets returned (70% chance per Sarah's report)
                if idx == 1:  # Second (larger) size
                    return_prob = 0.65  # 65% of time they keep the smaller
            
            # Seasonal modifier (higher returns in Jan after Christmas)
            if month == 1:
                return_prob += 0.05
            
            return_prob = np.clip(return_prob, 0, 1)
            
            # Determine if returned
            is_returned = random.random() < return_prob
            
            price = product['price']
            
            transactions.append({
                'order_id': f'ORD_{order_id}',
                'customer_id': customer_id,
                'product_id': product_id,
                'product_name': product['product_name'],
                'category': category,
                'size': size,
                'fit': product['fit'],
                'neckline': product['neckline'],
                'pattern': product['pattern'],
                'sleeve_style': product['sleeve_style'],
                'length': product['length'],
                'occasion': product['occasion'],
                'style_aesthetic': product['style_aesthetic'],
                'trend': product['trend'],
                'order_date': order_date,
                'order_month': order_date.strftime('%Y-%m'),
                'qty_purchased': 1,
                'qty_returned': 1 if is_returned else 0,
                'value_purchased': price,
                'value_returned': price if is_returned else 0,
                'is_multi_style_order': n_items > 1 and not is_multi_size,
                'is_multi_size_order': is_multi_size,
                'customer_segment': customer['segment'],
            })
            
            items_generated += 1
            if items_generated >= NUM_TRANSACTIONS:
                break
    
    return pd.DataFrame(transactions)


def main():
    print("=" * 60)
    print("MAPP FASHION - SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating customers...")
    customers_df = generate_customers()
    print(f"   Created {len(customers_df)} customers")
    print(f"   Segments: {customers_df['segment'].value_counts().to_dict()}")
    
    print("\n2. Generating products...")
    products_df = generate_products()
    print(f"   Created {len(products_df)} products")
    print(f"   Categories: {products_df['category'].value_counts().to_dict()}")
    
    print("\n3. Generating transactions...")
    transactions_df = generate_transactions(customers_df, products_df)
    print(f"   Created {len(transactions_df)} transactions")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    total_qty = transactions_df['qty_purchased'].sum()
    total_returns = transactions_df['qty_returned'].sum()
    overall_return_rate = total_returns / total_qty
    print(f"\nOverall Return Rate: {overall_return_rate:.1%}")
    print(f"Total Items Sold: {total_qty:,}")
    print(f"Total Items Returned: {total_returns:,}")
    
    print("\nReturn Rate by Category:")
    cat_stats = transactions_df.groupby('category').agg({
        'qty_purchased': 'sum',
        'qty_returned': 'sum'
    })
    cat_stats['return_rate'] = cat_stats['qty_returned'] / cat_stats['qty_purchased']
    cat_stats = cat_stats.sort_values('return_rate', ascending=False)
    for cat, row in cat_stats.iterrows():
        print(f"   {cat}: {row['return_rate']:.1%} ({int(row['qty_purchased']):,} items)")
    
    print("\nReturn Rate by Size (Numeric):")
    size_stats = transactions_df[transactions_df['size'].isin(NUMERIC_SIZES)].groupby('size').agg({
        'qty_purchased': 'sum',
        'qty_returned': 'sum'
    })
    size_stats['return_rate'] = size_stats['qty_returned'] / size_stats['qty_purchased']
    for size in NUMERIC_SIZES:
        if size in size_stats.index:
            row = size_stats.loc[size]
            print(f"   Size {size}: {row['return_rate']:.1%}")
    
    print("\nReturn Rate by Customer Segment:")
    seg_stats = transactions_df.groupby('customer_segment').agg({
        'qty_purchased': 'sum',
        'qty_returned': 'sum'
    })
    seg_stats['return_rate'] = seg_stats['qty_returned'] / seg_stats['qty_purchased']
    for seg, row in seg_stats.iterrows():
        print(f"   {seg}: {row['return_rate']:.1%}")
    
    # Save data
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    customers_df.to_csv(os.path.join(data_dir, 'customers.csv'), index=False)
    products_df.to_csv(os.path.join(data_dir, 'products.csv'), index=False)
    transactions_df.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False)
    
    print(f"\n✓ Data saved to {data_dir}")
    print("  - customers.csv")
    print("  - products.csv")
    print("  - transactions.csv")
    
    return customers_df, products_df, transactions_df


if __name__ == '__main__':
    main()
