import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel("large_bird_climate_asia_1980_2010.xlsx")

# Feature selection
features = ["Avg_Temperature_C", "Precipitation_mm", "Migration_Shift_km", "Bird_Traffic"]
target = "Population_Count"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

# Predict future impact
def simulate_future(df, model):
    simulated = []
    for _, row in df.iterrows():
        future_input = pd.DataFrame({
            "Avg_Temperature_C": [row["Avg_Temperature_C"] + 2],
            "Precipitation_mm": [row["Precipitation_mm"] - 100],
            "Migration_Shift_km": [row["Migration_Shift_km"] + 10],
            "Bird_Traffic": [row["Bird_Traffic"] + 200]
        })
        pred = model.predict(future_input)[0]
        simulated.append({
            "Species": row["Species"],
            "Year": row["Year"],
            "Population_Before": row["Population_Count"],
            "Population_After": pred
        })
    return pd.DataFrame(simulated)

simulated_df = simulate_future(df, model)
simulated_df.to_excel("simulated_future_bird_data.xlsx", index=False)

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

sns.lineplot(data=simulated_df, x="Year", y="Population_After", hue="Species", ax=axes[0, 0])
axes[0, 0].set_title("Projected Population Over Years")

sns.scatterplot(data=simulated_df, x="Population_Before", y="Population_After", hue="Species", ax=axes[0, 1])
axes[0, 1].set_title("Before vs After Climate Impact")

sns.boxplot(data=simulated_df, x="Species", y="Population_After", ax=axes[1, 0])
axes[1, 0].tick_params(axis='x', rotation=90)
axes[1, 0].set_title("Species-wise Population Prediction")

sns.heatmap(df[features + ["Population_Count"]].corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

sns.barplot(data=simulated_df.groupby("Species")["Population_After"].mean().reset_index(), x="Species", y="Population_After", ax=axes[2, 0])
axes[2, 0].tick_params(axis='x', rotation=90)
axes[2, 0].set_title("Average Predicted Population")

# Map Plot (Assumes lat/lon columns exist)
geo_df = gpd.GeoDataFrame(simulated_df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
world = gpd.read_file(r"C:\Users\PRATHMESH\Downloads\project\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp")
world_asia = world[world["CONTINENT"] == "Asia"]

world_asia.plot(figsize=(12, 8), color='lightgrey')
geo_df.plot(ax=plt.gca(), markersize=10, color='blue', alpha=0.5)
plt.title("Bird Population Impact Points in Asia")
plt.show()

# 10 Precautionary Actions
precautions = [
    "1. Establish climate-resilient bird sanctuaries.",
    "2. Implement stricter air pollution and deforestation controls.",
    "3. Monitor migratory routes with real-time satellite data.",
    "4. Support native vegetation growth to stabilize ecosystems.",
    "5. Use predictive models to pre-plan interventions.",
    "6. Increase public awareness on local bird protection.",
    "7. Encourage sustainable farming and water usage.",
    "8. Strengthen cross-border conservation treaties.",
    "9. Reduce artificial lighting during migration periods.",
    "10. Fund genetic diversity research and habitat restoration."
]

print("\nPrecautionary Actions to Prevent Bird Extinction:")
for item in precautions:
    print(item)
