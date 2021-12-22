import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

project_name = "mnist"
project = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(project, exist_ok=True)

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()
train_image_X = np.expand_dims(train_X, axis=-1)
train_X = np.reshape(train_X, (len(train_X), -1))

cm_bright = ListedColormap(
    ["#FF0000", "#0000FF", "#FFFF00", "#00FFFF", "#FF8800", "#0088FF", "#880000", "#000088", "#FF0088", "#8800FF"])

train_X_pca = MaxAbsScaler().fit_transform(PCA(n_components=2).fit_transform(train_X))

kmeans_labels = KMeans(n_clusters=10, random_state=0).fit_predict(train_X)


def compute_entropy(prob_distribution):
    log_prob = tf.math.log(prob_distribution + tf.keras.backend.epsilon())
    prob_log_prob = prob_distribution * log_prob
    entropy_val = -1.0 * tf.reduce_sum(prob_log_prob, axis=-1)
    return tf.reduce_mean(entropy_val)


def experiment(
        experiment_no,
        embedding_size,
        learning_rate,
        balance_coefficient,
):
    feature_extraction = tf.keras.Sequential(
        [
            tf.keras.layers.Input([28, 28, 1]),
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(embedding_size)
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    for epoch in range(1, 11):
        prog_bar = tqdm(range(len(train_image_X) // 100), desc=f"Training Epoch {epoch}")
        for step in prog_bar:
            global_step = ((epoch - 1) * len(train_image_X) // 100) + step + 1
            indices = np.random.choice(len(train_image_X), 100)
            image_batch, label_batch = train_image_X[indices], train_y[indices]
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

            if global_step % 600 == 0:
                logit = feature_extraction(train_image_X)
                p_n_given_x = tf.nn.softmax(logit, axis=-1)
                predictions = tf.argmax(p_n_given_x, axis=-1).numpy()

                fig, axes = plt.subplots(2, 2, figsize=(12, 12))

                axes[0, 0].set_xlabel("x[0]")
                axes[0, 0].set_ylabel("x[1]")
                axes[0, 0].title.set_text("Data")
                axes[0, 0].scatter(
                    train_X_pca[:, 0],
                    train_X_pca[:, 1],
                    c=train_y,
                    cmap=cm_bright,
                    edgecolors="k",
                )

                axes[1, 0].set_xlabel("x[0]")
                axes[1, 0].set_ylabel("x[1]")
                axes[1, 0].title.set_text("Unsupervised Information Gain Clustering")
                axes[1, 0].sharex(axes[0, 0])
                axes[1, 0].sharey(axes[0, 0])
                axes[1, 0].scatter(
                    train_X_pca[:, 0], train_X_pca[:, 1], c=predictions, cmap=cm_bright, edgecolors="k"
                )

                axes[0, 1].set_xlabel("x[0]")
                axes[0, 1].set_ylabel("x[1]")
                axes[0, 1].title.set_text("K-Means Clustering")
                axes[0, 1].sharex(axes[0, 0])
                axes[0, 1].sharey(axes[0, 0])
                axes[0, 1].scatter(
                    train_X_pca[:, 0], train_X_pca[:, 1], c=kmeans_labels, cmap=cm_bright, edgecolors="k"
                )

                logit = MaxAbsScaler().fit_transform(logit)
                axes[1, 1].set_xlabel("x[0]")
                axes[1, 1].set_ylabel("x[1]")
                axes[1, 1].title.set_text("Unsupervised Information Gain Clustering Logits")
                axes[1, 1].sharex(axes[0, 0])
                axes[1, 1].sharey(axes[0, 0])
                axes[1, 1].scatter(
                    logit[:, 0], logit[:, 1], c=train_y, cmap=cm_bright, edgecolors="k"
                )
                score = metrics.normalized_mutual_info_score(train_y, predictions)
                print(f"Unsupervised Deep Clustering NMI: {score * 100:.2f}% ")

                plt.savefig(f"{project}/experiment_{experiment_no}_step_{global_step}")

    return score


# cm_tab3 = cm.get_cmap("tab10")
# cm_list = cm_tab3(np.linspace(0, 1, 10))
# cm_classes = ListedColormap(cm_list)

if __name__ == "__main__":
    with open(f"{project}/experiments.csv", "w") as experiment_fh:
        experiment_csv = csv.DictWriter(
            experiment_fh,
            [
                "experiment_no",
                "learning_rate",
                "balance_coefficient",
                "score",
            ],
        )
        experiment_csv.writeheader()
        i = 0
        row = {
            "experiment_no": 'kmeans',
            "learning_rate": 0,
            "balance_coefficient": 0,
            "score": metrics.normalized_mutual_info_score(train_y, kmeans_labels)
        }

        experiment_csv.writerow(row)
        for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.1, 1]:
            for balance_coefficient in [1, 2, 5, 10]:
                experiment_config = {
                    "experiment_no": i,
                    "embedding_size": 10,
                    "learning_rate": learning_rate,
                    "balance_coefficient": balance_coefficient,

                }
                score = experiment(**experiment_config)
                row = {
                    "experiment_no": i,
                    "learning_rate": learning_rate,
                    "balance_coefficient": balance_coefficient,
                    "score": score
                }
                experiment_csv.writerow(row)
                i += 1
