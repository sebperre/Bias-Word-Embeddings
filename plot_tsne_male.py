import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path

# Load the data from the .dat file
file_path = path.join('top_association_clustering', 'tsne_clusters_male_male_vis_elkan_11.dat')
data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
data.columns = ['x', 'y', 'cluster', 'word']

print(data)

# Define cluster labels (adjust these based on your clusters)
cluster_labels = {
    0: 'Big Tech', #
    1: 'Engineering and Electronics', #
    2: 'Adventure and Music',
    3: 'Religion', #
    4: 'Numbers', #
    5: 'Sports', #
    6: 'Non-English Tokens', #
    7: 'Male Names', #
    8: 'Cities', #
    9: 'Cars', #
    10: 'Violence' #
}

# Map cluster numbers to labels
data['label'] = data['cluster'].map(cluster_labels)

# Define a color palette
palette = sns.color_palette("hsv", len(cluster_labels))

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='x', y='y',
    hue='label',
    palette=palette,
    data=data,
    legend='full',
    s=100  # size of points
)

# Customize plot (titles, labels, etc.)
plt.title('t-SNE visualization for Male Clusters')
plt.xlabel('T-SNE x-coordinate')
plt.ylabel('T-SNE y-coordinate')
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), title='Categories')
plt.grid(True)
plt.show()