import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.cluster import KMeans


plt.style.use("seaborn-v0_8-whitegrid")

# =========================
# LOAD
# =========================
#change path
df = pd.read_csv("NYC.csv")


# =========================
# FEATURE ENGINEERING
# =========================
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.dayofweek

def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["pickup_longitude"], df["pickup_latitude"],
    df["dropoff_longitude"], df["dropoff_latitude"]
)

df["expected_duration"] = df["distance_km"] * 120
df["expected_duration"] = df["expected_duration"].replace(0, 1)

df["delay_ratio"] = df["trip_duration"] / df["expected_duration"]
df["speed_kmh"] = df["distance_km"] / (df["trip_duration"] / 3600)

# =========================
# ZONE CREATION (GRID-BASED)
# =========================
df["zone_lat"] = df["pickup_latitude"].round(2)
df["zone_lon"] = df["pickup_longitude"].round(2)
df["zone"] = df["zone_lat"].astype(str) + "_" + df["zone_lon"].astype(str)

# =========================
# BAD ZONES DETECTION
# =========================
zone_perf = df.groupby("zone")["delay_ratio"].mean().sort_values(ascending=False)
bad_zones = zone_perf.head(5)

# =========================
# GEO CLUSTERING
# =========================
coords = df[["pickup_latitude", "pickup_longitude"]]
kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)
cluster_perf = df.groupby("cluster")["delay_ratio"].mean().sort_values(ascending=False)


# =========================
# ML MODEL
# =========================
features = ["distance_km", "hour", "day", "passenger_count"]
X = df[features]
y = df["trip_duration"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
model.fit(X_train, y_train)

df["predicted_duration"] = model.predict(X)
df["prediction_error"] = df["trip_duration"] - df["predicted_duration"]

def ai_delay_risk(row):
    if row["prediction_error"] > 300:
        return "High"
    elif row["prediction_error"] > 120:
        return "Medium"
    else:
        return "Low"

df["ai_delay_risk"] = df.apply(ai_delay_risk, axis=1)

mae = mean_absolute_error(y_test, model.predict(X_test))



# =========================
# CAPTAIN PERFORMANCE SCORE
# =========================
vendor_stats = df.groupby("vendor_id").agg({
    "delay_ratio": "mean",
    "prediction_error": "mean",
    "speed_kmh": "mean"
}).reset_index()
# Normalize score
vendor_stats["score"] = (
    (1 / vendor_stats["delay_ratio"]) * 0.5 +
    (1 / (vendor_stats["prediction_error"].abs() + 1)) * 0.3 +
    (vendor_stats["speed_kmh"] / vendor_stats["speed_kmh"].max()) * 0.2
)
vendor_stats = vendor_stats.sort_values(by="score", ascending=False)


# =========================
# KPIs
# =========================
avg_delay = round(df["delay_ratio"].mean(), 2)
high_delay = (df["ai_delay_risk"] == "High").sum()
avg_speed = round(df["speed_kmh"].mean(), 2)

# =========================
# CHARTS
# =========================
plt.figure(figsize=(8, 4))
delay_hour = df.groupby("hour")["delay_ratio"].mean()
plt.bar(delay_hour.index.astype(str), delay_hour.values)
plt.title("Peak Delay Hours (Operational Insight)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delay Ratio")
# value labels
for i, v in enumerate(delay_hour.values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("delay_hour.png", dpi=300)
plt.close()


# Delay Distribution 
plt.figure(figsize=(7, 4))
plt.hist(df["delay_ratio"], bins=50)
plt.title("Distribution of Trip Delays")
plt.xlabel("Delay Ratio")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("delay_distribution.png")
plt.close()

# =========================
# TOP 7 WORST TRIPS TABLE
# =========================
top_trips = df.sort_values(by="prediction_error", ascending=False).head(7)
table_data = [["ID", "Vendor", "Delay Ratio", "Prediction Deviation"]]

for _, row in top_trips.iterrows():
    table_data.append([
        row["id"],
        row["vendor_id"],
        round(row["delay_ratio"], 2),
        round(row["prediction_error"], 2)
    ])


# =========================
# ZONE MAP VISUALIZATION
# =========================
plt.figure(figsize=(6, 4))

# Color by cluster (or use bad zones)
plt.scatter(
    df["pickup_longitude"],
    df["pickup_latitude"],
    c=df["cluster"],
    cmap="viridis",
    s=5,
    alpha=0.5
)

# Highlight BAD ZONES (top 5)
for zone in bad_zones.index:
    lat, lon = zone.split("_")
    plt.scatter(float(lon), float(lat), color="red", s=80)

plt.title("Hotspots & Cluster Map (Pickup Locations)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("zone_map.png", dpi=300)
plt.close()


# =========================
# PDF GENERATION
# =========================
doc = SimpleDocTemplate("AI_Trip_Premium_Report.pdf")
styles = getSampleStyleSheet()
content = []

# Title
content.append(Paragraph("AI-Powered Trip Efficiency Report", styles["Title"]))
content.append(Spacer(1, 12))

# KPIs
content.append(Paragraph("Key Metrics", styles["Heading2"]))
# Average Delay Ratio
content.append(Paragraph(
    "Average Delay Ratio represents the overall efficiency of trips by measuring how much delay exists compared to expected time.",
    styles["Normal"]
))
content.append(Paragraph(
    f"<b>Value:</b> {round(avg_delay, 3)}",
    styles["Normal"]
))
# High Delay Trips
content.append(Paragraph(
    "High Delay Trips counts the number of trips that exceeded the acceptable delay threshold, highlighting operational issues and outliers.",
    styles["Normal"]
))
content.append(Paragraph(
    f"<b>Value:</b> {int(high_delay)}",
    styles["Normal"]
))

# Average Speed
content.append(Paragraph(
    "Average Speed indicates the overall movement efficiency of trips and reflects traffic conditions and routing performance.",
    styles["Normal"]
))
content.append(Paragraph(
    f"<b>Value:</b> {round(avg_speed, 2)} km/h",
    styles["Normal"]
))

# Model MAE
content.append(Paragraph(
    "Model MAE (Mean Absolute Error) measures the average prediction error of the model in seconds, indicating forecasting accuracy.",
    styles["Normal"]
))
content.append(Paragraph(
    f"<b>Value:</b> {round(mae, 2)} seconds",
    styles["Normal"]
))
content.append(Spacer(1, 12))

# Charts
content.append(Paragraph("Peak Delay Hours", styles["Heading2"]))
content.append(Image("delay_hour.png", width=400, height=200))
content.append(Paragraph(
    "This chart shows average delay ratio per hour of the day. "
    "Higher values indicate peak congestion periods where trip efficiency drops significantly.",
    styles["Normal"]
))
content.append(Spacer(1, 12))


# Table
content.append(Paragraph("Top 7 Problematic Trips", styles["Heading2"]))
table = Table(table_data)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 7),
    ("GRID", (0, 0), (-1, -1), 1, colors.black)
]))
content.append(table)
content.append(Spacer(1, 12))

#Detect worst zones
content.append(Paragraph("High Delay Zones (Critical Areas)", styles["Heading2"]))
for zone, val in bad_zones.items():
    content.append(Paragraph(f"Zone {zone}: Delay Ratio {round(val,2)}", styles["Normal"]))
content.append(Spacer(1, 12))

#Geeographical Clusters Performance
content.append(Paragraph("Geographical Clusters Performance", styles["Heading2"]))
for cluster, val in cluster_perf.items():
    content.append(Paragraph(f"Cluster {cluster}: Delay Ratio {round(val,2)}", styles["Normal"]))
content.append(Spacer(1, 12))
content.append(Paragraph("Geographical Hotspots & Clusters", styles["Heading2"]))
content.append(Image("zone_map.png", width=300, height=200))
content.append(Paragraph(
    "This map highlights high-demand and high-delay zones using clustering and grid-based segmentation. "
    "Red areas indicate critical zones requiring route or supply optimization.",
    styles["Normal"])
)
content.append(Spacer(1, 12))


#Captain Performance Ranking
content.append(Paragraph("Captain Performance Ranking", styles["Heading2"]))
for _, row in vendor_stats.iterrows():
    content.append(Paragraph(
        f"Vendor {row['vendor_id']} → Score: {round(row['score'],2)}",
        styles["Normal"]
    ))
content.append(Paragraph(
    "This ranking system evaluates vendors based on delay efficiency, prediction accuracy, and speed performance. "
    "Higher scores represent more reliable and efficient captains.",
    styles["Normal"])
)
content.append(Spacer(1, 12))

# Recommendations
content.append(Paragraph("AI Recommendations", styles["Heading2"]))
content.append(Paragraph("- Increase incentives during peak delay hours", styles["Normal"]))
content.append(Paragraph("- Optimize routing for congested zones", styles["Normal"]))
content.append(Paragraph("- Rebalance captain supply", styles["Normal"]))

# Build
doc.build(content)

print("✅ Premium PDF Generated: AI_Trip_Premium_Report.pdf")