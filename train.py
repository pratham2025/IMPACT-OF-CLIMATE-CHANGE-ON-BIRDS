import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel("bird_climate_impact_large.xlsx")

# Selecting features and target variable
features = ["Avg_Temperature_C", "Precipitation_mm", "Migration_Shift_km"]
target = "Population_Count"

X = df[features]
y = df[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-Squared Score: {r2}")

# Predict population decline for each bird species due to climate change
def predict_population_change(df, model):
    predictions = []
    for _, row in df.iterrows():
        input_data = pd.DataFrame({
            "Avg_Temperature_C": [row["Avg_Temperature_C"] + 2],  # Simulating climate change
            "Precipitation_mm": [row["Precipitation_mm"] - 100],
            "Migration_Shift_km": [row["Migration_Shift_km"] + 5]
        })
        predicted_population = model.predict(input_data)[0]
        predictions.append({
            "Species": row["Species"],
            "Year": row["Year"],
            "Population_Before": row["Population_Count"],
            "Population_After": predicted_population
        })
    return pd.DataFrame(predictions)

# Applying prediction function
predicted_df = predict_population_change(df, model)
predicted_df.to_excel("predicted_population_impact.xlsx", index=False)

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

sns.barplot(data=predicted_df, x="Species", y="Population_After", errorbar=None, ax=axes[0, 0])
axes[0, 0].set_title("Predicted Population of Each Species After Climate Change")
axes[0, 0].tick_params(axis='x', rotation=90)

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=axes[0, 1])
axes[0, 1].set_title("Correlation Heatmap of Climate Factors")

sns.violinplot(data=predicted_df, x="Species", y="Population_After", ax=axes[1, 0])
axes[1, 0].set_title("Distribution of Predicted Population After Climate Change")
axes[1, 0].tick_params(axis='x', rotation=90)

sns.lineplot(data=predicted_df, x="Year", y="Population_After", hue="Species", ax=axes[1, 1])
axes[1, 1].set_title("Trend of Population Changes Over the Years")

sns.scatterplot(data=predicted_df, x="Population_Before", y="Population_After", hue="Species", ax=axes[2, 0])
axes[2, 0].set_title("Comparison of Population Before and After Climate Change")

sns.boxplot(data=predicted_df, x="Species", y="Population_After", ax=axes[2, 1])
axes[2, 1].set_title("Boxplot of Predicted Population After Climate Change")
axes[2, 1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

print("Predictions saved to predicted_population_impact.xlsx")
