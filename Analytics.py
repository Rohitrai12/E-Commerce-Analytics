"""
E-Commerce Analytics:
==================================================
Python -> NumPy -> Pandas -> Visualization -> Data Quality ->
Feature Engineering -> Probability -> Statistics -> SQL preparation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

try:
    import duckdb
except ImportError:
    duckdb = None


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TABLE_FILES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "categories": "categories.csv",
    "sellers": "sellers.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments.csv",
    "shipments": "shipments.csv",
    "returns": "returns.csv",
    "reviews": "reviews.csv",
}


# ============================================================
# 2. DATA INGESTION
# ============================================================

def load_csv(filename: str) -> pd.DataFrame:
    """Load and validate a CSV file."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"{filename} exists but contains no rows.")

    return df


def load_all_tables() -> Dict[str, pd.DataFrame]:
    """Load every expected dataset into a dictionary."""
    return {
        name: load_csv(filename)
        for name, filename in TABLE_FILES.items()
    }


tables = load_all_tables()

customers = tables["customers"]
products = tables["products"]
categories = tables["categories"]
sellers = tables["sellers"]
orders = tables["orders"]
order_items = tables["order_items"]
payments = tables["payments"]
shipments = tables["shipments"]
returns = tables["returns"]
reviews = tables["reviews"]


# ============================================================
# 3. BASIC DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

for name, df in tables.items():
    print(f"{name:15} rows={len(df):>10,} columns={len(df.columns):>4}")


# ============================================================
# 4. DATA QUALITY AUDIT
# ============================================================

def audit_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return schema, missingness and uniqueness metrics."""
    return pd.DataFrame({
        "table": name,
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_pct": np.round(df.isna().mean().values * 100, 2),
        "unique_count": df.nunique(dropna=False).values,
    })


schema_audit = pd.concat(
    [audit_dataframe(df, name) for name, df in tables.items()],
    ignore_index=True,
)

print("\nMissing-value summary:")
print(
    schema_audit
    .query("missing_count > 0")
    .sort_values(["missing_pct", "table"], ascending=False)
    .head(30)
    .to_string(index=False)
)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Summarize duplicate rows."""
    duplicate_count = int(df.duplicated().sum())
    return {
        "rows": len(df),
        "duplicate_rows": duplicate_count,
        "duplicate_pct": round(duplicate_count / max(len(df), 1) * 100, 2),
    }


duplicate_summary = pd.DataFrame(
    duplicate_report(df) | {"table": name}
    for name, df in tables.items()
).set_index("table")

print("\nDuplicate summary:")
print(duplicate_summary)


# ============================================================
# 5. DATETIME STANDARDIZATION
# ============================================================

DATE_COLUMNS = {
    "customers": ["signup_date"],
    "orders": ["order_date"],
    "shipments": ["shipped_at", "delivered_at", "expected_delivery_at"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
}

for table_name, columns in DATE_COLUMNS.items():
    df = tables[table_name]

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )


# ============================================================
# 6. REFERENTIAL INTEGRITY CHECKS
# ============================================================

def validate_foreign_key(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
) -> pd.DataFrame:
    """Return child rows whose foreign key is absent in parent."""
    if child_column not in child_df.columns:
        return pd.DataFrame()

    if parent_column not in parent_df.columns:
        return pd.DataFrame()

    return child_df[
        ~child_df[child_column].isin(parent_df[parent_column])
    ]


integrity_checks = {
    "orders.customer_id -> customers.customer_id": validate_foreign_key(
        orders, "customer_id", customers, "customer_id"
    ),
    "order_items.order_id -> orders.order_id": validate_foreign_key(
        order_items, "order_id", orders, "order_id"
    ),
    "order_items.product_id -> products.product_id": validate_foreign_key(
        order_items, "product_id", products, "product_id"
    ),
    "products.category_id -> categories.category_id": validate_foreign_key(
        products, "category_id", categories, "category_id"
    ),
    "products.seller_id -> sellers.seller_id": validate_foreign_key(
        products, "seller_id", sellers, "seller_id"
    ),
}

print("\nReferential integrity checks:")
for relationship, invalid_rows in integrity_checks.items():
    print(f"{relationship}: {len(invalid_rows):,} invalid rows")


# ============================================================
# 7. SAFE CLEANING
# ============================================================

# Work on copies so raw tables remain available.
orders = orders.copy()
order_items = order_items.copy()
products = products.copy()
customers = customers.copy()
shipments = shipments.copy()
returns = returns.copy()
reviews = reviews.copy()

# Numeric coercion for common analytical fields.
NUMERIC_COLUMNS = {
    "orders": ["subtotal", "discount", "shipping_fee", "tax", "total_amount"],
    "order_items": ["quantity", "unit_price", "cost_price", "discount_pct"],
    "shipments": [],
    "reviews": [],
}

for table_name, columns in NUMERIC_COLUMNS.items():
    df = locals()[table_name]

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


# Remove exact duplicate records.
for name in ["orders", "order_items", "products", "customers", "shipments", "returns", "reviews"]:
    df = locals()[name]
    before = len(df)
    df = df.drop_duplicates().copy()
    locals()[name] = df


# ============================================================
# 8. FEATURE ENGINEERING: ORDERS
# ============================================================

if "order_date" in orders.columns:
    orders["order_year"] = orders["order_date"].dt.year
    orders["order_month"] = orders["order_date"].dt.month
    orders["order_quarter"] = orders["order_date"].dt.quarter
    orders["order_week"] = orders["order_date"].dt.isocalendar().week.astype("Int64")
    orders["order_day"] = orders["order_date"].dt.day
    orders["day_name"] = orders["order_date"].dt.day_name()
    orders["is_weekend"] = orders["order_date"].dt.dayofweek >= 5

if {"subtotal", "shipping_fee", "tax"}.issubset(orders.columns):
    orders["gross_revenue"] = (
        orders["subtotal"]
        + orders["shipping_fee"]
        + orders["tax"]
    )

if {"gross_revenue", "discount"}.issubset(orders.columns):
    orders["net_revenue"] = (
        orders["gross_revenue"]
        - orders["discount"].fillna(0)
    )
elif "total_amount" in orders.columns:
    orders["net_revenue"] = orders["total_amount"]


# ============================================================
# 9. FEATURE ENGINEERING: ORDER ITEMS
# ============================================================

required_item_columns = {
    "quantity",
    "unit_price",
}

if required_item_columns.issubset(order_items.columns):
    order_items["gross_sales"] = (
        order_items["quantity"] * order_items["unit_price"]
    )

if {"gross_sales", "discount_pct"}.issubset(order_items.columns):
    order_items["discount_amount"] = (
        order_items["gross_sales"]
        * order_items["discount_pct"].fillna(0)
        / 100
    )
else:
    order_items["discount_amount"] = 0.0

if "gross_sales" in order_items.columns:
    order_items["net_sales"] = (
        order_items["gross_sales"]
        - order_items["discount_amount"]
    )

if {"quantity", "cost_price"}.issubset(order_items.columns):
    order_items["cost"] = (
        order_items["quantity"] * order_items["cost_price"]
    )

if {"net_sales", "cost"}.issubset(order_items.columns):
    order_items["gross_profit"] = (
        order_items["net_sales"] - order_items["cost"]
    )

if {"net_sales", "gross_profit"}.issubset(order_items.columns):
    order_items["gross_margin_pct"] = np.where(
        order_items["net_sales"] > 0,
        order_items["gross_profit"] / order_items["net_sales"] * 100,
        np.nan,
    )


# ============================================================
# 10. NUMPY STATISTICAL PROFILE
# ============================================================

if "net_revenue" in orders.columns:
    order_values = orders["net_revenue"].dropna().to_numpy(dtype=float)

    metrics = {
        "mean": np.mean(order_values),
        "median": np.median(order_values),
        "std": np.std(order_values),
        "variance": np.var(order_values),
        "min": np.min(order_values),
        "max": np.max(order_values),
        "p25": np.percentile(order_values, 25),
        "p75": np.percentile(order_values, 75),
        "p90": np.percentile(order_values, 90),
        "p95": np.percentile(order_values, 95),
        "p99": np.percentile(order_values, 99),
    }

    order_distribution = pd.Series(metrics, name="net_revenue")

    print("\nOrder value statistical profile:")
    print(order_distribution)


# ============================================================
# 11. MONTHLY SALES ANALYTICS
# ============================================================

if {"order_date", "net_revenue", "order_id", "customer_id"}.issubset(orders.columns):
    monthly_sales = (
        orders
        .groupby(orders["order_date"].dt.to_period("M"))
        .agg(
            revenue=("net_revenue", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
            avg_order_value=("net_revenue", "mean"),
        )
        .reset_index()
    )

    monthly_sales["month"] = monthly_sales["order_date"].astype(str)
    monthly_sales["mom_growth_pct"] = monthly_sales["revenue"].pct_change() * 100
    monthly_sales["rolling_3m_revenue"] = (
        monthly_sales["revenue"].rolling(3, min_periods=1).mean()
    )
else:
    monthly_sales = pd.DataFrame()


# ============================================================
# 12. CUSTOMER ANALYTICS
# ============================================================

if {"customer_id", "order_id", "net_revenue", "order_date"}.issubset(orders.columns):
    customer_metrics = (
        orders
        .groupby("customer_id")
        .agg(
            total_orders=("order_id", "nunique"),
            total_spend=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .reset_index()
    )

    customer_metrics["customer_lifetime_days"] = (
        customer_metrics["last_order"]
        - customer_metrics["first_order"]
    ).dt.total_seconds() / 86400

    customer_metrics["is_repeat_customer"] = (
        customer_metrics["total_orders"] > 1
    )
else:
    customer_metrics = pd.DataFrame()


# ============================================================
# 13. RFM SEGMENTATION
# ============================================================

if not customer_metrics.empty:
    analysis_date = orders["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        orders
        .groupby("customer_id")
        .agg(
            recency=(
                "order_date",
                lambda x: (analysis_date - x.max()).days,
            ),
            frequency=("order_id", "nunique"),
            monetary=("net_revenue", "sum"),
        )
        .reset_index()
    )

    rfm["R_score"] = pd.qcut(
        rfm["recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["M_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["RFM_score"] = (
        rfm["R_score"]
        + rfm["F_score"]
        + rfm["M_score"]
    )

    def classify_customer(score: int) -> str:
        if score >= 13:
            return "Champions"
        if score >= 10:
            return "Loyal Customers"
        if score >= 7:
            return "Potential Loyalists"
        if score >= 5:
            return "At Risk"
        return "Lost Customers"

    rfm["segment"] = rfm["RFM_score"].apply(classify_customer)
else:
    rfm = pd.DataFrame()


# ============================================================
# 14. PRODUCT ANALYTICS
# ============================================================

if {"product_id", "quantity", "net_sales", "gross_profit"}.issubset(order_items.columns):
    product_performance = (
        order_items
        .groupby("product_id")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("net_sales", "sum"),
            profit=("gross_profit", "sum"),
            avg_unit_price=("unit_price", "mean"),
        )
        .reset_index()
    )

    product_columns = [
        col for col in [
            "product_id",
            "category_id",
            "seller_id",
            "brand",
            "product_name",
        ]
        if col in products.columns
    ]

    if product_columns:
        product_performance = product_performance.merge(
            products[product_columns].drop_duplicates("product_id"),
            on="product_id",
            how="left",
        )

    product_performance["margin_pct"] = np.where(
        product_performance["revenue"] > 0,
        product_performance["profit"]
        / product_performance["revenue"]
        * 100,
        np.nan,
    )
else:
    product_performance = pd.DataFrame()


# ============================================================
# 15. RETURN ANALYTICS
# ============================================================

if {"product_id", "quantity"}.issubset(returns.columns) and not product_performance.empty:
    return_metrics = (
        returns
        .groupby("product_id")
        .agg(
            returned_orders=("order_id", "nunique")
            if "order_id" in returns.columns else ("product_id", "size"),
            return_units=("quantity", "sum"),
        )
        .reset_index()
    )

    product_performance = product_performance.merge(
        return_metrics,
        on="product_id",
        how="left",
    )

    for col in ["returned_orders", "return_units"]:
        product_performance[col] = product_performance[col].fillna(0)

    product_performance["return_rate_pct"] = np.where(
        product_performance["units_sold"] > 0,
        product_performance["return_units"]
        / product_performance["units_sold"]
        * 100,
        np.nan,
    )


# ============================================================
# 16. DELIVERY ANALYTICS
# ============================================================

if {"shipped_at", "delivered_at"}.issubset(shipments.columns):
    shipments["delivery_hours"] = (
        shipments["delivered_at"]
        - shipments["shipped_at"]
    ).dt.total_seconds() / 3600

if {"delivered_at", "expected_delivery_at"}.issubset(shipments.columns):
    shipments["is_late"] = (
        shipments["delivered_at"]
        > shipments["expected_delivery_at"]
    )

if "is_late" in shipments.columns:
    late_delivery_rate = shipments["is_late"].mean() * 100
    print(f"\nLate delivery rate: {late_delivery_rate:.2f}%")


# ============================================================
# 17. PROBABILITY ANALYSIS
# ============================================================

if "order_id" in orders.columns and "order_id" in returns.columns:
    total_orders = orders["order_id"].nunique()
    returned_orders = returns["order_id"].nunique()

    if total_orders:
        p_return = returned_orders / total_orders
        print(f"P(Return) = {p_return:.4f}")

    if {"order_id", "is_late"}.issubset(shipments.columns):
        late_orders = set(
            shipments.loc[shipments["is_late"], "order_id"]
        )
        on_time_orders = set(
            shipments.loc[~shipments["is_late"], "order_id"]
        )
        returned_order_set = set(returns["order_id"])

        if late_orders:
            p_return_given_late = (
                len(late_orders & returned_order_set)
                / len(late_orders)
            )
            print(
                f"P(Return | Late Delivery) = "
                f"{p_return_given_late:.4f}"
            )

        if on_time_orders:
            p_return_given_on_time = (
                len(on_time_orders & returned_order_set)
                / len(on_time_orders)
            )
            print(
                f"P(Return | On-time Delivery) = "
                f"{p_return_given_on_time:.4f}"
            )


# ============================================================
# 18. HYPOTHESIS TESTING
# ============================================================

if (
    {"order_id", "net_revenue"}.issubset(orders.columns)
    and {"order_id", "is_late"}.issubset(shipments.columns)
):
    late_order_ids = set(
        shipments.loc[shipments["is_late"], "order_id"]
    )

    late_aov = orders.loc[
        orders["order_id"].isin(late_order_ids),
        "net_revenue",
    ].dropna()

    on_time_aov = orders.loc[
        ~orders["order_id"].isin(late_order_ids),
        "net_revenue",
    ].dropna()

    if len(late_aov) >= 2 and len(on_time_aov) >= 2:
        statistic, p_value = ttest_ind(
            late_aov,
            on_time_aov,
            equal_var=False,
            nan_policy="omit",
        )

        print("\nHypothesis test: Late vs On-time AOV")
        print(f"t-statistic = {statistic:.4f}")
        print(f"p-value     = {p_value:.6f}")

        alpha = 0.05
        if p_value < alpha:
            print("Result: statistically significant at alpha=0.05")
        else:
            print("Result: not statistically significant at alpha=0.05")


# ============================================================
# 19. CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    col for col in [
        "net_revenue",
        "subtotal",
        "discount",
        "shipping_fee",
        "tax",
    ]
    if col in orders.columns
]

if len(correlation_columns) >= 2:
    correlation_matrix = orders[correlation_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix)


# ============================================================
# 20. VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")

if not monthly_sales.empty:
    plt.figure(figsize=(13, 6))
    plt.plot(
        monthly_sales["month"],
        monthly_sales["revenue"],
        marker="o",
        label="Monthly Revenue",
    )
    plt.plot(
        monthly_sales["month"],
        monthly_sales["rolling_3m_revenue"],
        linewidth=3,
        label="3M Rolling Average",
    )
    plt.xticks(rotation=45)
    plt.title("Monthly Revenue and 3-Month Rolling Average")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_revenue.png", dpi=160)
    plt.show()

if "net_revenue" in orders.columns:
    plt.figure(figsize=(11, 6))
    sns.histplot(
        orders["net_revenue"].dropna(),
        bins=60,
        kde=True,
    )
    plt.title("Distribution of Order Value")
    plt.xlabel("Net Revenue per Order")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "order_value_distribution.png", dpi=160)
    plt.show()

if not rfm.empty:
    plt.figure(figsize=(10, 6))
    segment_counts = rfm["segment"].value_counts()
    sns.barplot(
        x=segment_counts.index,
        y=segment_counts.values,
    )
    plt.xticks(rotation=30, ha="right")
    plt.title("Customer Segments by RFM")
    plt.xlabel("Customer Segment")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rfm_segments.png", dpi=160)
    plt.show()

if not product_performance.empty:
    top_products = product_performance.nlargest(10, "revenue")

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_products,
        x="revenue",
        y="product_id",
    )
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_products.png", dpi=160)
    plt.show()

if len(correlation_columns) >= 2:
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        square=True,
    )
    plt.title("E-Commerce Financial Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.show()


# ============================================================
# 21. SQL ANALYTICS WITH DUCKDB
# ============================================================

SQL_QUERIES = {
    "monthly_revenue": """
        SELECT
            DATE_TRUNC('month', order_date) AS month,
            SUM(net_revenue) AS revenue,
            COUNT(DISTINCT order_id) AS orders,
            COUNT(DISTINCT customer_id) AS customers,
            AVG(net_revenue) AS avg_order_value
        FROM orders
        GROUP BY 1
        ORDER BY 1;
    """,

    "top_products": """
        SELECT
            product_id,
            SUM(net_sales) AS revenue,
            SUM(gross_profit) AS profit,
            SUM(quantity) AS units_sold
        FROM order_items
        GROUP BY product_id
        ORDER BY revenue DESC
        LIMIT 20;
    """,

    "customer_ranking": """
        SELECT
            customer_id,
            SUM(net_revenue) AS lifetime_value,
            COUNT(DISTINCT order_id) AS total_orders,
            RANK() OVER (
                ORDER BY SUM(net_revenue) DESC
            ) AS customer_rank
        FROM orders
        GROUP BY customer_id
        ORDER BY customer_rank;
    """,

    "running_revenue": """
        SELECT
            order_date,
            net_revenue,
            SUM(net_revenue) OVER (
                ORDER BY order_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_revenue
        FROM orders
        ORDER BY order_date;
    """,

    "monthly_growth": """
        WITH monthly AS (
            SELECT
                DATE_TRUNC('month', order_date) AS month,
                SUM(net_revenue) AS revenue
            FROM orders
            GROUP BY 1
        )
        SELECT
            month,
            revenue,
            LAG(revenue) OVER (ORDER BY month) AS previous_month,
            ROUND(
                100.0 * (
                    revenue - LAG(revenue) OVER (ORDER BY month)
                )
                / NULLIF(
                    LAG(revenue) OVER (ORDER BY month),
                    0
                ),
                2
            ) AS mom_growth_pct
        FROM monthly
        ORDER BY month;
    """,

    "category_performance": """
        SELECT
            c.category_name,
            SUM(oi.net_sales) AS revenue,
            SUM(oi.gross_profit) AS profit,
            SUM(oi.quantity) AS units_sold
        FROM order_items oi
        JOIN products p
            ON oi.product_id = p.product_id
        JOIN categories c
            ON p.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY revenue DESC;
    """,
}

if duckdb is not None:
    con = duckdb.connect()

    # Register transformed DataFrames so SQL runs against the same analytical layer.
    for name, df in {
        "customers": customers,
        "products": products,
        "categories": categories,
        "sellers": sellers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "shipments": shipments,
        "returns": returns,
        "reviews": reviews,
    }.items():
        con.register(name, df)

    sql_results = {}

    for query_name, query in SQL_QUERIES.items():
        try:
            sql_results[query_name] = con.execute(query).df()
            print(f"\nSQL query completed: {query_name}")
            print(sql_results[query_name].head(10))
        except Exception as exc:
            print(f"\nSQL query failed: {query_name}\n{exc}")

    con.close()
else:
    sql_results = {}
    print(
        "\nDuckDB is not installed. Install it with: pip install duckdb"
    )


# ============================================================
# 22. EXPORT ANALYTICAL OUTPUTS
# ============================================================

if not schema_audit.empty:
    schema_audit.to_csv(
        OUTPUT_DIR / "schema_audit.csv",
        index=False,
    )

if not customer_metrics.empty:
    customer_metrics.to_csv(
        OUTPUT_DIR / "customer_metrics.csv",
        index=False,
    )

if not rfm.empty:
    rfm.to_csv(
        OUTPUT_DIR / "rfm_segmentation.csv",
        index=False,
    )

if not product_performance.empty:
    product_performance.to_csv(
        OUTPUT_DIR / "product_performance.csv",
        index=False,
    )

if not monthly_sales.empty:
    monthly_sales.to_csv(
        OUTPUT_DIR / "monthly_sales.csv",
        index=False,
    )

for query_name, result in sql_results.items():
    result.to_csv(
        OUTPUT_DIR / f"sql_{query_name}.csv",
        index=False,
    )


# ============================================================
# 23. EXECUTIVE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

if "net_revenue" in orders.columns:
    print(f"Total revenue: ${orders['net_revenue'].sum():,.2f}")

if "order_id" in orders.columns:
    print(f"Total orders: {orders['order_id'].nunique():,}")

if "customer_id" in orders.columns:
    print(f"Unique customers: {orders['customer_id'].nunique():,}")

if "net_revenue" in orders.columns and "order_id" in orders.columns:
    aov = orders.groupby("order_id")["net_revenue"].sum().mean()
    print(f"Average order value: ${aov:,.2f}")

if "is_repeat_customer" in customer_metrics.columns:
    repeat_rate = customer_metrics["is_repeat_customer"].mean() * 100
    print(f"Repeat customer rate: {repeat_rate:.2f}%")

if "is_late" in shipments.columns:
    print(f"Late delivery rate: {shipments['is_late'].mean() * 100:.2f}%")

if "return_id" in returns.columns or not returns.empty:
    print(f"Returned records: {len(returns):,}")

print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
print("Analysis complete.")
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_16/1191329247.py in <cell line: 0>()
     72 
     73 
---> 74 tables = load_all_tables()
     75 
     76 customers = tables["customers"]

/tmp/ipykernel_16/1191329247.py in load_all_tables()
     67     """Load every expected dataset into a dictionary."""
     68     return {
---> 69         name: load_csv(filename)
     70         for name, filename in TABLE_FILES.items()
     71     }

/tmp/ipykernel_16/1191329247.py in load_csv(filename)
     54 
     55     if not path.exists():
---> 56         raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
     57 
     58     df = pd.read_csv(path, low_memory=False)

FileNotFoundError: Dataset not found: /kaggle/working/data/customers.csv
