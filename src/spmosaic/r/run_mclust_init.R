args <- commandArgs(trailingOnly = TRUE)

embedding_csv <- args[1]
labels_csv <- args[2]
centers_csv <- args[3]
n_clusters <- as.integer(args[4])
model_name <- args[5]
seed <- as.integer(args[6])

set.seed(seed)

suppressPackageStartupMessages({library(mclust)})

X <- read.csv(embedding_csv, header = TRUE, check.names = FALSE)
X_mat <- as.matrix(X)

fit <- Mclust(X_mat, G = n_clusters, modelNames = model_name)

labels <- data.frame(cluster = fit$classification - 1)
write.csv(labels, labels_csv, row.names = FALSE)

y_pred <- fit$classification - 1
unique_labels <- sort(unique(y_pred))
centers <- do.call(rbind, lapply(unique_labels, function(lbl) {
  colMeans(X_mat[y_pred == lbl, , drop = FALSE])
}))
centers <- as.data.frame(centers)
write.csv(centers, centers_csv, row.names = FALSE)