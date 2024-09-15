"""
Visualize embeddings using PCA, t-SNE, UMAP, KMeans clustering and other.
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
import umap
from wordcloud import WordCloud

from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)


class VisualizeEmbeddings:
    def __init__(
        self,
        db: Chroma,
        method: str = "pca",
        n_clusters: int = 5,
        n_components: int = 2,
        sample_size: int = 100,
        max_clusters: int = 10,
    ) -> None:
        """
        Initialize the Visualization object.

        Args:
            db (Chroma): The Chroma database.
            method (str): Dimensionality reduction method ("pca" or "tsne")
            n_clusters (int): Number of clusters for KMeans clustering.
        """
        self.logger = logging.getLogger(__name__)

        self.db = db
        self.method = method
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.sample_size = sample_size
        self.max_clusters = max_clusters

        self.embeddings = []
        self.documents = []
        self.metadatas = []

        self.reduced_embeddings = []
        self.labels = []

    def _load_embeddings(self) -> None:
        """
        Load embeddings from the Chroma database.
        """
        data = self.db.get(include=["embeddings", "documents", "metadatas"])
        self.embeddings = np.array(data["embeddings"])
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

    def _reduce_dimensions(self) -> None:
        """
        Reduce the dimensions of the embeddings using PCA or t-SNE.
        """
        if self.method == "pca":
            pca = PCA(n_components=self.n_components)
            self.reduced_embeddings = pca.fit_transform(self.embeddings)
            self.logger.info(
                f"PCA: Explained variance ratio for {self.n_components} components: {pca.explained_variance_ratio_}"
            )
        elif self.method == "tsne":
            tsne = TSNE(n_components=self.n_components, perplexity=30, random_state=42)
            self.reduced_embeddings = tsne.fit_transform(self.embeddings)
        else:
            self.logger.error("Invalid method. Please choose 'pca' or 'tsne'.")
            exit(1)

    def _cluster_embeddings(self) -> None:
        """
        Cluster the embeddings using KMeans.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(self.embeddings)

    def _visualize_clusters(self, output_path: str) -> None:
        """
        Visualize the embeddings.

        Args:
            output_path (str): The path to save the visualization.
        """
        df = pd.DataFrame(self.reduced_embeddings, columns=["x", "y"])
        df["label"] = self.labels

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="viridis")
        plt.title(f"{self.method.upper()} Visualization of Embeddings")

        plt.savefig(output_path)
        self.logger.info(f"Visualization saved to {output_path}")

    def _visualize_density(self, output_path: str) -> None:
        """
        Visualize the density of the embeddings.

        Args:
            output_path (str): The path to save the visualization.
        """
        df = pd.DataFrame(self.reduced_embeddings, columns=["x", "y"])

        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=df, x="x", y="y", fill=True, cmap="viridis")
        plt.title(f"{self.method.upper()} Density of Embeddings")

        plt.savefig(output_path)
        self.logger.info(f"Visualization saved to {output_path}")

    def _visualize_umap(self, output_path: str) -> None:
        """
        Visualize the embeddings using UMAP.

        Args:
            output_path (str): The path to save the visualization.
        """
        reducer = umap.UMAP(n_components=2)
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)

        df = pd.DataFrame(self.reduced_embeddings, columns=["x", "y"])
        df["label"] = self.labels

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x="x", y="y", palette="viridis", hue="label")
        plt.title("UMAP Visualization of Embeddings")

        plt.savefig(output_path)
        self.logger.info(f"Visualization saved to {output_path}")

    def _visualize_similarity(self, output_path: str) -> None:
        """
        Visualize the similarity between embeddings.

        Args:
            output_path (str): The path to save the visualization.
        """
        if len(self.embeddings) > self.sample_size:
            idx = np.random.choice(
                len(self.embeddings), self.sample_size, replace=False
            )
            sampled_embeddings = self.embeddings[idx]
        else:
            sampled_embeddings = self.embeddings

        similarity = cosine_similarity(sampled_embeddings)
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity, cmap="viridis")
        plt.title("Cosine Similarity between Embeddings")

        plt.savefig(output_path)
        self.logger.info(f"Visualization saved to {output_path}")

    def _visualize_silhouette(self, output_path: str) -> None:
        """
        Visualize silhouette analysis for clustering.

        Args:
            output_path (str): The path to save the silhouette analysis.
        """
        silhouette_avg = silhouette_score(self.embeddings, self.labels)
        self.logger.info(f"Average silhouette score: {silhouette_avg}")

        sample_silhouette_values = silhouette_samples(self.embeddings, self.labels)
        plt.figure(figsize=(12, 8))

        y_lower = 10
        for i in range(self.n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[self.labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = sns.color_palette("viridis", as_cmap=True)(
                float(i) / self.n_clusters
            )
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            y_lower = y_upper + 10

        plt.title(f"Silhouette analysis for {self.n_clusters} clusters")
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster")

        plt.savefig(output_path)
        self.logger.info(f"Silhouette analysis saved to {output_path}")

    def _visualize_wordcloud(self, output_path: str) -> None:
        """
        Visualize word clouds for each cluster.

        Args:
            output_path (str): The path to save the word cloud visualizations.
        """
        for i in range(self.n_clusters):
            plt.figure(figsize=(12, 8))

            cluster_text = " ".join(
                [text for text, label in zip(self.documents, self.labels) if label == i]
            )

            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(cluster_text)

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud for Cluster {i}")

            plt.savefig(f"{output_path}_cluster_{i}.png")
            self.logger.info(
                f"Word Cloud for cluster {i} saved to {output_path}_cluster_{i}.png"
            )

    def _visualize_elbow(self, output_path: str) -> None:
        """
        Visualize the elbow method for KMeans clustering.

        Args:
            output_path (str): The path to save the visualization.
        """
        distortions = []
        K = range(2, self.max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.embeddings)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(12, 8))
        plt.plot(K, distortions, "bo-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        plt.title("Elbow Method for Optimal Number of Clusters")
        plt.grid(True)
        plt.savefig(output_path)
        self.logger.info(f"Elbow method visualization saved to {output_path}")

    def _visualize_cluster_composition(
        self, output_path: str, metadata_key: str = "source"
    ) -> None:
        """
        Visualize the composition of the clusters.

        Args:
            output_path (str): The path to save the visualization.
            metadata_key (str): The metadata key to use for composition.
        """
        metadata_values = [
            doc_metadata.get(metadata_key, "Unknown") for doc_metadata in self.metadatas
        ]

        df = pd.DataFrame({"cluster": self.labels, "metadata": metadata_values})

        plt.figure(figsize=(12, 8))
        sns.countplot(x="cluster", hue="metadata", data=df, palette="Set2")
        plt.title(f"Cluster Composition by Metadata by {metadata_key}")
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.legend(title="Source Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.savefig(output_path)
        self.logger.info(f"Cluster composition saved to {output_path}")

    def _visualize_intra_cluster_similarity(self, output_path: str) -> None:
        """
        Visualize intra-cluster similarity using cosine similarity.

        Args:
            output_path (str): The path to save the visualization.
        """
        intra_cluster_similarities = []
        for cluster in range(self.n_clusters):
            cluster_embeddings = self.embeddings[self.labels == cluster]
            if len(cluster_embeddings) > 1:
                sim_matrix = cosine_similarity(cluster_embeddings)
                avg_similarity = np.mean(
                    sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
                )
                intra_cluster_similarities.append(avg_similarity)
            else:
                intra_cluster_similarities.append(1)
        plt.figure(figsize=(12, 8))
        plt.bar(range(self.n_clusters), intra_cluster_similarities, color="skyblue")
        plt.title("Average Intra-Cluster Similarity")
        plt.xlabel("Cluster")
        plt.ylabel("Average Cosine Similarity")
        plt.savefig(output_path)
        self.logger.info(
            f"Intra-cluster similarity visualization saved to {output_path}"
        )

    def visualize(
        self,
        output_dir: str,
    ) -> None:
        """
        Visualize the embeddings.

        Args:
            output_dir (str): The directory to save the visualizations.
        """
        self._load_embeddings()
        self._reduce_dimensions()
        self._cluster_embeddings()

        self._visualize_clusters(
            os.path.join(
                output_dir, f"embeddings_{self.method}_clusters_{self.n_clusters}.png"
            )
        )
        self._visualize_density(
            os.path.join(output_dir, f"embeddings_{self.method}_density.png")
        )
        self._visualize_umap(
            os.path.join(output_dir, f"embeddings_umap_clusters_{self.n_clusters}.png")
        )
        self._visualize_similarity(
            os.path.join(
                output_dir, f"embeddings_similarity_size_{self.sample_size}.png"
            )
        )
        self._visualize_silhouette(
            os.path.join(output_dir, f"embeddings_silhouette_{self.n_clusters}.png")
        )

        self._visualize_elbow(
            os.path.join(output_dir, f"embeddings_elbow_{self.max_clusters}.png")
        )
        self._visualize_cluster_composition(
            os.path.join(
                output_dir,
                f"embeddings_composition_clusters_{self.n_clusters}_key_source.png",
            ),
        )
        self._visualize_intra_cluster_similarity(
            os.path.join(
                output_dir, f"embeddings_intra_similarity_{self.n_clusters}.png"
            )
        )

        self._visualize_wordcloud(
            os.path.join(
                output_dir, "wordcloud", f"embeddings_clusters_{self.n_clusters}"
            )
        )

        self.logger.info("All visualizations saved.")
