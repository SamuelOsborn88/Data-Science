import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths MAKE SURE TO CHANGE THE PATHS IF NEEDED WHEN SHOWCASING ON PERSONAL PC
# ALSO YOU MAY NEED TO MAKE A NEW DATASET FILE IN PERSONAL
input_path = r"C:\Users\V979429\Documents\Sam Stuffs\Test\retail_price_volume.csv"
output_dir = r"C:\Users\V979429\Documents\Sam Stuffs\Test\test data and charts"

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)


# Load Prep
df = pd.read_csv(input_path)

df["week"] = pd.to_datetime(df["week"])
df = df.sort_values(by=["brand", "week"])

# Week over Week changes
df["pct_price_change"] = df.groupby("brand")["price"].pct_change()
df["pct_units_change"] = df.groupby("brand")["units_sold"].pct_change()


# Elasticity
df["elasticity"] = df["pct_units_change"] / df["pct_price_change"]


# Classification
def classify(e):
    if pd.isna(e):
        return None
    elif abs(e) > 1:
        return "Elastic"
    else:
        return "Inelastic"

df["elasticity_type"] = df["elasticity"].apply(classify)


# Output Table
elasticity_summary = (
    df.groupby("brand")["elasticity"]
    .mean()
    .reset_index()
    .rename(columns={"elasticity": "avg_elasticity"})
)

print("\n=== Avg Elasticity by Brand ===")
print(elasticity_summary)


# Save Results
df.to_csv(os.path.join(output_dir, "output_with_elasticity.csv"), index=False)
elasticity_summary.to_csv(os.path.join(output_dir, "elasticity_summary.csv"), index=False)


# Chart: Price vs Units Trend
for brand in df["brand"].unique():
    subset = df[df["brand"] == brand]
    plt.figure()
    plt.plot(subset["week"], subset["price"], label="Price")
    plt.plot(subset["week"], subset["units_sold"], label="Units Sold")
    plt.title(f"{brand} - Price vs Units Trend")
    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    file_name = f"{brand.replace(' ', '_')}_trend.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


# Scatter Plot
plt.figure()
plt.scatter(df["price"], df["units_sold"])
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.title("Price vs Units Sold (All Brands)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_vs_units_scatter.png"))
plt.close()


# Done
print("\nDone. Files generated in:")
print(output_dir)