"""
AI-Driven Store Footfall Impact Analysis on Sales
STAGE 1 - Data Understanding, Cleaning, and Total_Sales creation

Dataset : ccount.csv  (Dominick's Finer Foods "Customer Count" file, Kilts Center)
Output  : data/clean_footfall_sales.csv  (used by every later stage)

Run:  python stage1_data_understanding.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/ccount.csv"
CLEAN_PATH = "data/clean_footfall_sales.csv"


# ----------------------------------------------------------------------
# 1. COLUMN DICTIONARY
# ----------------------------------------------------------------------
# Department DOLLAR SALES columns. Verified to be peer departments (not
# nested inside one another), so summing them does not double count.
SALES_COLS = [
    "grocery", "dairy", "frozen", "bottle", "mvpclub",
    "meat", "meatfroz", "fish",
    "produce", "bulk", "saladbar",
    "deli", "deliself", "deliexpr", "convfood", "cheese",
    "bakery", "pharmacy", "floral", "promo",
    "gm", "jewelry", "cosmetic", "haba",
    "camera", "photofin", "video", "videoren",
    "beer", "wine", "spirits",
]

# COUPONS REDEEMED (stored as negative dollars). NOT revenue - kept separate.
COUPON_COLS = [
    "groccoup", "meatcoup", "fishcoup", "promcoup", "prodcoup", "bulkcoup",
    "salcoup", "florcoup", "delicoup", "pharcoup", "gmcoup", "vidcoup",
    "miscscp",
]

# Columns that are 100% zero in every row of this file - dropped.
DEAD_COLS = [
    "ftgchin", "ftgccoup", "ftgital", "ftgicoup", "daircoup", "frozcoup",
    "habacoup", "photcoup", "cosmcoup", "ssdelicp", "bakcoup", "liqcoup",
]

# Ambiguous: named like a coupon but always positive. Excluded from revenue.
AMBIGUOUS_COLS = ["mancoup"]

FOOTFALL_COL = "custcoun"


# ----------------------------------------------------------------------
# 2. LOAD  (encoding matters - without latin-1 the date column is corrupted)
# ----------------------------------------------------------------------
def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, dtype={"date": "object"},
                     encoding="latin-1", low_memory=False)
    print(f"Raw shape: {df.shape}")
    return df


# ----------------------------------------------------------------------
# 3. INSPECTION
# ----------------------------------------------------------------------
def inspect(df):
    print("\n--- dtypes ---")
    print(df.dtypes.value_counts())

    print("\n--- missing values (top 10) ---")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss[miss > 0].head(10))

    print("\n--- all-zero columns ---")
    zero = [c for c in df.columns
            if df[c].dtype != object and df[c].fillna(0).eq(0).all()]
    print(zero)

    print("\n--- sales vs coupon signature (% of rows negative) ---")
    for c in SALES_COLS[:5] + COUPON_COLS[:5]:
        label = "SALES " if c in SALES_COLS else "COUPON"
        print(f"  {label} {c:10s} {(df[c] < 0).mean() * 100:5.1f}%")

    print("\n--- nesting test: are sub-departments inside parents? ---")
    for parent, kids in [("deli", ["deliself", "deliexpr", "convfood", "cheese"]),
                         ("gm", ["jewelry", "cosmetic", "haba", "camera",
                                 "photofin", "video", "videoren"])]:
        over = (df[kids].sum(axis=1) > df[parent]).mean() * 100
        print(f"  {parent}: children exceed parent in {over:.1f}% of rows "
              f"-> {'NOT nested (safe to sum)' if over > 1 else 'possibly nested'}")


# ----------------------------------------------------------------------
# 4. CLEANING
# ----------------------------------------------------------------------
def clean(df):
    n0 = len(df)
    log = []

    def step(name):
        nonlocal n0
        dropped = n0 - len(df)
        log.append(f"{name:38s} removed {dropped:7,d}  remaining {len(df):7,d}")
        n0 = len(df)

    # 4.1 Parse date. Values are YYMMDD integers, e.g. 900331 -> 1990-03-31.
    df["date_num"] = pd.to_numeric(df["date"], errors="coerce")
    df = df[df["date_num"].between(880000, 999999)].copy()
    step("drop corrupted / unparseable dates")

    df["date"] = pd.to_datetime(df["date_num"].astype(int).astype(str),
                                format="%y%m%d", errors="coerce")
    df = df[df["date"].notna()]
    step("drop rows where date failed to parse")

    # 4.2 Keep 1989 onward. 1988 is a partial start-up period.
    df = df[df["date"].dt.year >= 1989]
    step("drop pre-1989 start-up period")

    # 4.3 Drop dead and unused columns
    df = df.drop(columns=DEAD_COLS + AMBIGUOUS_COLS + ["week", "date_num"],
                 errors="ignore")

    # 4.4 Missing values. Rows missing footfall are unusable (that is the
    #     variable being studied). Missing department values mean the
    #     department recorded nothing that day, so 0 is the correct fill.
    df = df[df[FOOTFALL_COL].notna()]
    step("drop rows with missing footfall")
    df[SALES_COLS] = df[SALES_COLS].fillna(0)
    df[COUPON_COLS] = df[COUPON_COLS].fillna(0)

    # 4.5 Duplicates: one row per store per day is the intended grain
    df = df.drop_duplicates(subset=["store", "date"], keep="first")
    step("drop duplicate store-day rows")

    # 4.6 Build Total_Sales BEFORE outlier filtering so the filter can use it
    df["Total_Sales"] = df[SALES_COLS].sum(axis=1)
    df["Total_Coupons"] = df[COUPON_COLS].sum(axis=1).abs()
    df["Net_Sales"] = df["Total_Sales"] - df["Total_Coupons"]

    # 4.7 Invalid / implausible values
    df = df[df["Total_Sales"] > 0]
    step("drop non-positive Total_Sales")

    df = df[df[FOOTFALL_COL].between(100, 20000)]
    step("drop implausible footfall (<100 or >20000)")

    df = df[df["Total_Sales"] < 300_000]
    step("drop extreme sales outliers")

    # 4.8 Drop stores with too little history to analyse
    counts = df["store"].value_counts()
    df = df[df["store"].isin(counts[counts >= 100].index)]
    step("drop stores with under 100 days of data")

    # 4.9 Time features for later stages
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["dow_num"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow_num"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 4.10 Useful derived metric
    df["Sales_Per_Customer"] = df["Total_Sales"] / df[FOOTFALL_COL]

    print("\n--- cleaning log ---")
    for line in log:
        print(" ", line)

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 5. VALIDATION
# ----------------------------------------------------------------------
def validate(df):
    print("\n--- cleaned dataset ---")
    print(f"Rows            : {len(df):,}")
    print(f"Stores          : {df['store'].nunique()}")
    print(f"Date range      : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Mean footfall   : {df[FOOTFALL_COL].mean():,.0f} customers/day")
    print(f"Mean Total_Sales: ${df['Total_Sales'].mean():,.0f}/day")
    print(f"Sales/customer  : ${df['Sales_Per_Customer'].median():,.2f} (median)")
    print(f"Correlation footfall vs Total_Sales: "
          f"{df[FOOTFALL_COL].corr(df['Total_Sales']):.4f}")

    # Sanity check: a realistic supermarket basket is roughly $15-$40.
    basket = df["Sales_Per_Customer"].median()
    verdict = "PASS" if 10 < basket < 60 else "FAIL - check SALES_COLS"
    print(f"\nBasket-size sanity check: {verdict}")

    print("\n--- department share of Total_Sales (top 10) ---")
    share = (df[SALES_COLS].sum() / df["Total_Sales"].sum() * 100)
    print(share.sort_values(ascending=False).head(10).round(2).to_string())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw()
    inspect(raw)
    clean_df = clean(raw)
    validate(clean_df)
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")