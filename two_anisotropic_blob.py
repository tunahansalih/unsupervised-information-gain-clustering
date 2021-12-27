import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn import datasets, metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, MaxAbsScaler
from tqdm import tqdm

project_name = "two_anisotropic_blob"
project = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(project, exist_ok=True)

X, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=2, shuffle=True, random_state=170, cluster_std=0.7)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

cm_bright = ListedColormap(["#FF0000", "#0000FF"])


def compute_entropy(prob_distribution):
    log_prob = tf.math.log(prob_distribution + tf.keras.backend.epsilon())
    prob_log_prob = prob_distribution * log_prob
    entropy_val = -1.0 * tf.reduce_sum(prob_log_prob)
    return tf.reduce_mean(entropy_val)


kmeans_labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)


def experiment(
        experiment_no,
        learning_rate=0.1,
        balance_coefficient=1,
):
    k = 2

    feature_extraction = tf.keras.Sequential(
        [
            tf.keras.layers.Input(2),
            # tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(k),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(1, 11):
        prog_bar = tqdm(range(len(X) // 100), desc=f"Training Epoch {epoch}")
        for step in prog_bar:
            global_step = ((epoch - 1) * len(X) // 100) + step + 1
            indices = np.random.choice(len(X), 100)
            image_batch, label_batch = X[indices], y[indices]
            with tf.GradientTape() as tape:
                logit = feature_extraction(image_batch)
                p_n_given_x = tf.nn.softmax(logit, axis=-1)
                h_p_n_given_x = compute_entropy(p_n_given_x) / tf.cast(tf.shape(p_n_given_x)[0], tf.float32)
                p_n = tf.reduce_mean(p_n_given_x, axis=0)
                h_p_n = compute_entropy(p_n)

                loss = h_p_n_given_x - balance_coefficient * h_p_n

            prog_bar.set_postfix(
                {
                    "Total Loss": f"{loss.numpy().item():.8f}",
                    "H[p(n)]": f"{h_p_n.numpy().item():.8f}",
                    "H[p(n|x)]": f"{h_p_n_given_x.numpy().item():.8f}",
                }
            )

            grads = tape.gradient(loss, feature_extraction.trainable_variables)
            optimizer.apply_gradients(zip(grads, feature_extraction.trainable_variables))

            if global_step % 50 == 0:
                logit = feature_extraction(X)
                p_n_given_x = tf.nn.softmax(logit, axis=-1)
                predictions = tf.argmax(p_n_given_x, axis=-1).numpy()

                fig, axes = plt.subplots(2, 2, figsize=(12, 12))

                axes[0, 0].set_xlabel("x[0]")
                axes[0, 0].set_ylabel("x[1]")
                axes[0, 0].title.set_text("Data")
                axes[0, 0].scatter(
                    X[:, 0],
                    X[:, 1],
                    c=y,
                    cmap=cm_bright,
                    edgecolors="k",
                )

                axes[1, 0].set_xlabel("x[0]")
                axes[1, 0].set_ylabel("x[1]")
                axes[1, 0].title.set_text("Unsupervised Information Gain Clustering")
                axes[1, 0].sharex(axes[0, 0])
                axes[1, 0].sharey(axes[0, 0])
                axes[1, 0].scatter(
                    X[:, 0], X[:, 1], c=predictions, cmap=cm_bright, edgecolors="k"
                )

                axes[0, 1].set_xlabel("x[0]")
                axes[0, 1].set_ylabel("x[1]")
                axes[0, 1].title.set_text("K-Means Clustering")
                axes[0, 1].sharex(axes[0, 0])
                axes[0, 1].sharey(axes[0, 0])
                axes[0, 1].scatter(
                    X[:, 0], X[:, 1], c=kmeans_labels, cmap=cm_bright, edgecolors="k"
                )

                logit = MaxAbsScaler().fit_transform(logit)
                axes[1, 1].set_xlabel("x[0]")
                axes[1, 1].set_ylabel("x[1]")
                axes[1, 1].title.set_text("Unsupervised Information Gain Clustering Logits")
                axes[1, 1].sharex(axes[0, 0])
                axes[1, 1].sharey(axes[0, 0])
                axes[1, 1].scatter(
                    logit[:, 0], logit[:, 1], c=y, cmap=cm_bright, edgecolors="k"
                )
                score_nmi = metrics.normalized_mutual_info_score(y, predictions)
                score_rand = metrics.rand_score(y, predictions)
                print(f"NMI: {score_nmi * 100:2f}% Rand: {score_rand * 100:2f}%")
                plt.savefig(f"{project}/experiment_{experiment_no}_step_{global_step}")

    return score_nmi, score_rand

if __name__ == "__main__":
    with open(f"{project}/experiments.csv", "w") as experiment_fh:
        experiment_csv = csv.DictWriter(
            experiment_fh,
            [
                "experiment_no",
                "learning_rate",
                "balance_coefficient",
                "score_nmi",
                "score_rand",
            ],
        )
        experiment_csv.writeheader()
        i = 0

        score_nmi = metrics.normalized_mutual_info_score(y, kmeans_labels)
        score_rand = metrics.rand_score(y, kmeans_labels)
        row = {
            "experiment_no": i,
            "score_nmi": score_nmi,
            "score_rand": score_rand
        }
        print(f"NMI: {score_nmi * 100:2f}% Rand: {score_rand * 100:2f}%")

        experiment_csv.writerow(row)

        for learning_rate in [0.05]:
            for balance_coefficient in [1]:
                experiment_config = {
                    "experiment_no": i,
                    "learning_rate": learning_rate,
                    "balance_coefficient": balance_coefficient,

                }
                score_nmi, score_rand = experiment(**experiment_config)
                row = {
                    "experiment_no": i,
                    "learning_rate": learning_rate,
                    "balance_coefficient": balance_coefficient,
                    "score_nmi": score_nmi,
                    "score_rand": score_rand
                }
                experiment_csv.writerow(row)
                i += 1
