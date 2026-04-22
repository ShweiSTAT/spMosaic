"""PyTorch model definitions for spMosaic.

This module contains neural network components used in the spMosaic
pipeline, including:

- a dual-encoder autoencoder for disentangling biological signal and
  batch-related variation
- a DEC model for clustering refinement based on the biological embedding
"""

from __future__ import annotations

import torch 
import torch.nn as nn
import torch.nn.functional as F

class DualEncoderAutoencoder(nn.Module):
    """Dual-encoder autoencoder for batch-aware representation learning.

    This model learns two latent embeddings from the same input gene
    expression profile:

    - ``h_G``: a biological embedding intended to capture domain-related
      structure
    - ``h_B``: a batch embedding intended to capture sample-specific or
      technical variation

    The two embeddings are concatenated and passed through a decoder to
    reconstruct the input expression. In addition, the batch embedding
    ``h_B`` is fed into a batch classifier to predict batch labels.

    Parameters
    ----------
    input_dim : int
        Number of input features, typically the number of selected genes.
    embedding_dim : int
        Dimension of each latent embedding branch.
    batch_num : int
        Number of batch classes.

    Attributes
    ----------
    encoder_G : nn.Sequential
        Encoder network for the biological embedding.
    encoder_B : nn.Sequential
        Encoder network for the batch embedding.
    decoder : nn.Sequential
        Decoder network that reconstructs the input from the concatenated
        embeddings.
    batch_classifier : nn.Sequential
        Classifier that predicts batch labels from ``h_B``.
    """

    def __init__(self, input_dim, embedding_dim, batch_num):
        super(DualEncoderAutoencoder, self).__init__()

        # Encoder for biological information
        self.encoder_G = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, embedding_dim)
        )

        # Encoder for batch effect
        self.encoder_B = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, embedding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Batch classifier
        self.batch_classifier = nn.Sequential(
            nn.Linear(embedding_dim, batch_num),
            nn.ReLU(),
            nn.Linear(batch_num, batch_num)
        )

    def forward(self, x):
        """Run a forward pass of the dual-encoder autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(n_samples, input_dim)``.

        Returns
        -------
        x_reconstructed : torch.Tensor
            Reconstructed input expression.
        batch_pred : torch.Tensor
            Predicted batch logits from the batch embedding ``h_B``.
        h_G : torch.Tensor
            Biological embedding.
        h_B : torch.Tensor
            Batch embedding.
        """
        h_G = self.encoder_G(x)
        h_B = self.encoder_B(x)

        combined = torch.cat((h_G, h_B), dim=1)
        x_reconstructed = self.decoder(combined)
        batch_pred = self.batch_classifier(h_B)

        return x_reconstructed, batch_pred, h_G, h_B
    

def loss_function(x, x_reconstructed, batch_labels, batch_pred, fuse_weight):
    """Compute the stage 2 pretraining loss for the dual-encoder autoencoder.

    The total loss consists of two components:

    - a reconstruction loss between the original input and the decoder output
    - a batch-classification loss based on the predicted batch logits

    The final loss is defined as:

    .. math::
        L = L_{recon} + \\lambda L_{batch},

    where ``fuse_weight`` corresponds to :math:`\\lambda`.

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape ``(n_samples, input_dim)``.
    x_reconstructed : torch.Tensor
        Reconstructed input tensor of the same shape as ``x``.
    batch_labels : torch.Tensor
        True batch labels of shape ``(n_samples,)``.
    batch_pred : torch.Tensor
        Predicted batch logits of shape ``(n_samples, batch_num)``.
    fuse_weight : float
        Weight applied to the batch-classification loss.

    Returns
    -------
    recon_loss : torch.Tensor
        Mean squared error reconstruction loss.
    class_loss : torch.Tensor
        Cross-entropy batch classification loss.
    total_loss : torch.Tensor
        Weighted sum of reconstruction and classification losses.
    """
    recon_loss = F.mse_loss(x_reconstructed, x)
    class_loss = F.cross_entropy(batch_pred, batch_labels)
    return recon_loss, class_loss, recon_loss + fuse_weight * class_loss


class DEC(nn.Module):
    """Deep Embedded Clustering model for spatial domain refinement.

    This model performs clustering refinement using the biological embedding
    learned by a pretrained :class:`DualEncoderAutoencoder`. During DEC
    refinement:

    - the biological encoder ``encoder_G`` remains trainable
    - the batch encoder ``encoder_B`` is frozen
    - cluster centers are treated as trainable parameters
    - the decoder is reused to preserve reconstruction structure

    The model computes soft cluster assignments based on the distance between
    each biological embedding and the trainable cluster centers, and then
    derives the DEC auxiliary target distribution used for clustering
    refinement.

    Parameters
    ----------
    pretrained_autoencoder : DualEncoderAutoencoder
        Pretrained autoencoder that provides the encoders and decoder.
    embedding_dim : int
        Dimension of the biological embedding.
    num_clusters : int
        Number of clusters.
    cluster_centers : torch.Tensor
        Initial cluster centers of shape ``(num_clusters, embedding_dim)``.

    Attributes
    ----------
    encoder_G : nn.Module
        Trainable biological encoder copied from the pretrained autoencoder.
    encoder_B : nn.Module
        Frozen batch encoder copied from the pretrained autoencoder.
    cluster_centers : nn.Parameter
        Trainable cluster centroids.
    decoder : nn.Module
        Decoder copied from the pretrained autoencoder for reconstruction.
    """

    def __init__(self, pretrained_autoencoder, embedding_dim, num_clusters, cluster_centers):
        super(DEC, self).__init__()

        # Use pretrained encoder_G for biological information
        self.encoder_G = pretrained_autoencoder.encoder_G

        # Use pretrained encoder_B but freeze it (so H_B is fixed)
        self.encoder_B = pretrained_autoencoder.encoder_B
        for param in self.encoder_B.parameters():
            param.requires_grad = False

        # Cluster centroids (trainable)
        self.cluster_centers = nn.Parameter(cluster_centers)

        # Keep decoder for reconstruction loss
        self.decoder = pretrained_autoencoder.decoder

    def target_distribution(self, q):
        """Compute the DEC auxiliary target distribution.

        The target distribution emphasizes confident cluster assignments and is
        used to refine the soft assignment matrix during DEC training.

        Parameters
        ----------
        q : torch.Tensor
            Soft cluster assignment matrix of shape
            ``(n_samples, num_clusters)``.

        Returns
        -------
        torch.Tensor
            Target distribution of the same shape as ``q``.
        """
        weight = (q ** 2) / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()

    def forward(self, x):
        """Run a forward pass of the DEC model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(n_samples, input_dim)``.

        Returns
        -------
        h_G : torch.Tensor
            Biological embedding of shape ``(n_samples, embedding_dim)``.
        q : torch.Tensor
            Soft cluster assignment matrix of shape
            ``(n_samples, num_clusters)``.
        p : torch.Tensor
            DEC target distribution of the same shape as ``q``.
        x_reconstructed : torch.Tensor
            Reconstructed input expression obtained from the concatenated
            biological and frozen batch embeddings.
        """
        h_G = self.encoder_G(x)
        h_B = self.encoder_B(x)

        # Compute q: soft cluster assignment
        q = 1.0 / (1.0 + torch.sum((h_G.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)

        # Compute target distribution p
        p = self.target_distribution(q)

        # Reconstruct gene expression using fused embedding
        combined = torch.cat((h_G, h_B), dim=1)
        x_reconstructed = self.decoder(combined)

        return h_G, q, p, x_reconstructed