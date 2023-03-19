# %% read dataframe
import pandas as pd

df = pd.read_pickle("data.pkl")

# %% pick a crime
df_murder = df[df["detailcm"] == "MURDER"]

# %% apply hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()

# %%
for k in tqdm(range(num_city, num_pct, 15)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_murder[["lat", "lon"]])
    scores[k] = silhouette_score(df_murder[["lat", "lon"]], y)
    labels[k] = y

# %% find the best k visually
import seaborn as sns

sns.lineplot(x=scores.keys(), y=scores.values())


# %% find the best k by code
best_k = max(scores, key=lambda k: scores[k])


# %% visualize the hierarchical clustering result
import folium

m = folium.Map((40.7128, -74.0060))
colors = sns.color_palette("hls", best_k).as_hex()
df_murder["label"] = labels[best_k]
for r in df_murder.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m


# %% find reason for stop columns
# and apply dbscan
from sklearn.cluster import DBSCAN

css = [col for col in df.columns if col.startswith("cs_")]
c = DBSCAN()
x = df_murder[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))


# %% visualize the result on map
import numpy as np

m = folium.Map((40.7128, -74.0060))
k = len(np.unique(y))
colors = sns.color_palette("hls", k).as_hex()
df_murder["label"] = y
for r in df_murder.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m

# %%
df_murder["label"].value_counts()


# %% pick a label and visualize the datapoints on map
biggest_cluster = df_murder["label"].value_counts().index[0]
m = folium.Map((40.7128, -74.0060))
for r in df_murder[df_murder["label"] == 0].to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m

# %%
