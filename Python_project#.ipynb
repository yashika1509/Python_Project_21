import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION (Update if column names differ after running Task 1)
# ==========================================
FILE_PATH = "21_retail_customer_orders.csv"
PRODUCT_COL = "Product_Name"      # e.g., 'Product' or 'Product_Name'
CATEGORY_COL = "Category"         # e.g., 'Category' or 'Product_Category'
PAYMENT_COL = "Payment_Method"    # e.g., 'Payment_Method' or 'Payment_Type'

# ==========================================
# TASK 1: Load CSV & Inspect Dimensions, Columns, Data Types
# ==========================================
df = pd.read_csv('/21_retail_customer_orders.csv')

print("--- TASK 1: DATASET INSPECTION ---")
print(f"Shape (Rows, Columns): {df.shape}")
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())

# ==========================================
# TASK 2: Identify Missing Values & Duplicate Order_IDs
# ==========================================
print("\n--- TASK 2: MISSING VALUES & DUPLICATES ---")
print("Missing values per column:\n", df.isnull().sum())

duplicate_orders = df.duplicated(subset=["Order_ID"]).sum()
print(f"\nDuplicate Order_ID records: {duplicate_orders}")

# Remove duplicate Order_IDs (keeping the first occurrence)
df = df.drop_duplicates(subset=["Order_ID"], keep="first")

# ==========================================
# TASK 3: Handle Missing Quantity & Unit_Price
# ==========================================
# Strategy: Fill missing Quantity and Unit_Price with their respective medians.
# Median is robust against extreme outliers compared to the mean.
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())

print("\n--- TASK 3: MISSING VALUES AFTER CLEANING ---")
print(df[["Quantity", "Unit_Price"]].isnull().sum())

# ==========================================
# TASK 4: Convert Order_Date & Extract Month, Weekday, Year
# ==========================================
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month_name()
df["Month_Num"] = df["Order_Date"].dt.month
df["Weekday"] = df["Order_Date"].dt.day_name()

print("\n--- TASK 4: EXTRACTED DATE FEATURES ---")
print(df[["Order_Date", "Year", "Month", "Weekday"]].head())

# ==========================================
# TASK 5 & 6: Create Order_Value & Calculate Totals
# ==========================================
df["Order_Value"] = df["Quantity"] * df["Unit_Price"]

total_orders = df["Order_ID"].nunique()
total_units_sold = df["Quantity"].sum()
total_revenue = df["Order_Value"].sum()

print("\n--- TASK 5 & 6: OVERALL SALES METRICS ---")
print(f"Total Orders:        {total_orders}")
print(f"Total Units Sold:    {total_units_sold:,.0f}")
print(f"Total Sales Revenue: {total_revenue:,.2f}")

# ==========================================
# TASK 7: Top 10 Products by Sales Revenue
# ==========================================
top_10_products = (
    df.groupby(PRODUCT_COL)["Order_Value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
print("\n--- TASK 7: TOP 10 PRODUCTS BY REVENUE ---")
print(top_10_products)

# ==========================================
# TASK 8: Rank Product Categories by Quantity & Revenue
# ==========================================
category_summary = (
    df.groupby(CATEGORY_COL)
    .agg(Total_Quantity=("Quantity", "sum"), Total_Revenue=("Order_Value", "sum"))
    .sort_values(by="Total_Revenue", ascending=False)
    .reset_index()
)
print("\n--- TASK 8: CATEGORY RANKING ---")
print(category_summary)

# ==========================================
# TASK 9: Average Order Value Across Payment Methods
# ==========================================
payment_aov = (
    df.groupby(PAYMENT_COL)["Order_Value"]
    .mean()
    .sort_values(ascending=False)
    .reset_index(name="Average_Order_Value")
)
print("\n--- TASK 9: AVERAGE ORDER VALUE BY PAYMENT METHOD ---")
print(payment_aov)

# ==========================================
# TASK 10: Busiest Shopping Days & Peak Sales Month
# ==========================================
busiest_days = df["Weekday"].value_counts()
monthly_sales = (
    df.groupby(["Month_Num", "Month"])["Order_Value"]
    .sum()
    .reset_index()
    .sort_values(by="Month_Num")
)
peak_month = monthly_sales.loc[monthly_sales["Order_Value"].idxmax()]

print("\n--- TASK 10: BUSIEST DAYS & PEAK MONTH ---")
print("Orders by Weekday:\n", busiest_days)
print("\nMonthly Sales Trend:\n", monthly_sales[["Month", "Order_Value"]])
print(f"\nPeak Sales Month: {peak_month['Month']} (Revenue: {peak_month['Order_Value']:,.2f})")

# ==========================================
# TASK 11: Dictionary Summary & Set of Unique Categories
# ==========================================
category_sales_dict = df.groupby(CATEGORY_COL)["Order_Value"].sum().round(2).to_dict()
unique_categories_set = set(df[CATEGORY_COL].dropna().unique())

print("\n--- TASK 11: DICTIONARY & SET ---")
print("Category Sales Dictionary:", category_sales_dict)
print("Unique Categories Set:", unique_categories_set)

# ==========================================
# TASK 12: Function to Get Total Sales by Category
# ==========================================
def get_category_sales(category_name, data=df):
    """Returns total sales revenue for a given category from the cleaned DataFrame."""
    filtered = data[data[CATEGORY_COL].str.lower() == category_name.lower()]
    if filtered.empty:
        return f"Category '{category_name}' not found."
    return round(filtered["Order_Value"].sum(), 2)

print("\n--- TASK 12: CATEGORY SALES FUNCTION TEST ---")
sample_cat = next(iter(unique_categories_set))
print(f"Total sales for '{sample_cat}': {get_category_sales(sample_cat)}")

# ==========================================
# TASK 13: NumPy Statistics for Order_Value
# ==========================================
order_values = df["Order_Value"].to_numpy()
mean_val = np.mean(order_values)
median_val = np.median(order_values)
std_val = np.std(order_values)

print("\n--- TASK 13: ORDER_VALUE STATISTICS (NUMPY) ---")
print(f"Mean:               {mean_val:,.2f}")
print(f"Median:             {median_val:,.2f}")
print(f"Standard Deviation: {std_val:,.2f}")

# ==========================================
# TASK 14: IQR Outlier Detection & Order-Value Bands
# ==========================================
q1 = np.percentile(order_values, 25)
q3 = np.percentile(order_values, 75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

outliers = df[df["Order_Value"] > upper_bound]
print("\n--- TASK 14: OUTLIERS & ORDER-VALUE BANDS ---")
print(f"IQR Upper Threshold: {upper_bound:,.2f}")
print(f"Number of unusually high orders: {len(outliers)}")
print(outliers[["Order_ID", PRODUCT_COL, "Order_Value"]].head(10))

# Create Low, Medium, High bands using 33rd and 66th percentiles
low_thresh = np.percentile(order_values, 33.3)
high_thresh = np.percentile(order_values, 66.6)

df["Order_Value_Band"] = pd.cut(
    df["Order_Value"],
    bins=[-np.inf, low_thresh, high_thresh, np.inf],
    labels=["Low", "Medium", "High"]
)
print("\nOrder Value Band Distribution:\n", df["Order_Value_Band"].value_counts())

# Save cleaned dataset (as recommended in project hints)
df.to_csv("21_retail_customer_orders_cleaned.csv", index=False)
print("\nCleaned dataset saved as '21_retail_customer_orders_cleaned.csv'.")

# ==========================================
# TASK 15: Product Class & 5 Sales Recommendations
# ==========================================
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def display_info(self):
        return f"Product: {self.name} | Unit Price: {self.price:.2f}"

print("\n--- TASK 15: PRODUCT CLASS DEMO ---")
sample_row = df.iloc[0]
prod_obj = Product(sample_row[PRODUCT_COL], sample_row["Unit_Price"])
print(prod_obj.display_info())

recommendations = [
    "1. Prioritize inventory and ad spend on the top 10 revenue-generating products to prevent stockouts.",
    "2. Bundle low-performing categories with top-selling items to lift average order value (AOV).",
    "3. Run targeted flash sales on slower weekdays to smooth out weekly order volume.",
    "4. Prepare extra stock and marketing campaigns 3–4 weeks ahead of the peak sales month.",
    "5. Promote the highest-AOV payment methods (e.g., via small cashback or EMI perks) to encourage larger baskets."
]

print("\n--- FIVE RECOMMENDATIONS FOR IMPROVING SALES ---")
for rec in recommendations:
    print(rec)
