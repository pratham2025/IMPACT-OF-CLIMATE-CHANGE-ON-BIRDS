import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Bird Climate Impact Dashboard", layout="wide")
st.title("🕊️ Bird Population & Climate Impact Dashboard")

# Load data
data_file = "large_bird_climate_asia_1980_2010.xlsx"
shape_file = "C:\\Users\\PRATHMESH\\Downloads\\project\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp"

@st.cache_data
def load_data():
    df = pd.read_excel(data_file)
    return df

def load_map():
    gdf = gpd.read_file(shape_file)
    return gdf[gdf['CONTINENT'] == 'Asia']

# Load datasets
df = load_data()
asia_map = load_map()

# Sidebar options
species = st.sidebar.multiselect("Select Bird Species:", options=df["Species"].unique(), default=df["Species"].unique())
years = st.sidebar.slider("Select Year Range:", int(df["Year"].min()), int(df["Year"].max()), (1990, 2010))

filtered_df = df[(df["Species"].isin(species)) & (df["Year"].between(years[0], years[1]))]

# Show dataset
with st.expander("📊 Show Raw Data"):
    st.dataframe(filtered_df)

# Graph 1: Population Over Time
st.subheader("📈 Bird Population Over Time")
fig1 = px.line(filtered_df, x="Year", y="Population_Count", color="Species", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# Graph 2: Climate Factors vs Population
st.subheader("🌡️ Climate Factors vs Bird Population")
fig2 = px.scatter(filtered_df, x="Avg_Temperature_C", y="Population_Count", color="Species", size="Bird_Traffic", hover_data=["Year"])
st.plotly_chart(fig2, use_container_width=True)

# Graph 3: Migration Impact
st.subheader("🛫 Migration Shift Impact")
fig3 = px.bar(filtered_df, x="Species", y="Migration_Shift_km", color="Species")
st.plotly_chart(fig3, use_container_width=True)

# Graph 4: Bird Traffic Distribution
st.subheader("🚦 Bird Traffic by Species")
fig4 = px.box(filtered_df, x="Species", y="Bird_Traffic", color="Species")
st.plotly_chart(fig4, use_container_width=True)

# Asia Map
st.subheader("🗺️ Map of Asia")
m = folium.Map(location=[20, 100], zoom_start=3)
folium.GeoJson(asia_map).add_to(m)
folium_static(m)

# ML Prediction
st.subheader("🤖 Predict Future Bird Population")
features = ["Avg_Temperature_C", "Precipitation_mm", "Migration_Shift_km", "Bird_Traffic"]
X = df[features]
y = df["Population_Count"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
st.markdown(f"**Model MAE:** {mae:.2f}")

# Simulate future scenario
future_data = df.copy()
future_data["Avg_Temperature_C"] += 2
future_data["Precipitation_mm"] -= 100
future_data["Migration_Shift_km"] += 5
future_preds = model.predict(future_data[features])
future_data["Future_Population"] = future_preds

fig5 = px.line(future_data, x="Year", y="Future_Population", color="Species", title="Projected Bird Population")
st.plotly_chart(fig5, use_container_width=True)

# Precautions Section
st.subheader("🛡️ 10 Precautions to Prevent Bird Extinction")
precautions = [
    "1. Preserve bird migration habitats and wetlands",
    "2. Implement stricter air pollution control",
    "3. Ban harmful pesticides impacting bird health",
    "4. Reforest and increase green zones",
    "5. Create urban bird sanctuaries",
    "6. Promote climate-resilient agricultural practices",
    "7. Educate public about bird conservation",
    "8. Fund climate-bird research projects",
    "9. Reduce light pollution in migration zones",
    "10. Enforce anti-poaching and illegal trade laws"
]
st.write("\n".join(precautions))

st.success("Dashboard Ready! Explore and Predict Bird Trends 🐦")
