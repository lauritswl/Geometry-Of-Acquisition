from typing import Optional
import numpy as np

"""
Provides a small utility class to build a concept vector from two sets
of embeddings (source / target examples). The concept vector is
by default the difference between the mean source and mean target
embeddings, optionally L2-normalized. You can project new embeddings
onto the concept vector or compute similarity scores.
"""

class ConceptVector:
    """
    Build and use a concept vector from two embedding sets.

    Usage:
        cv = ConceptVector()
        cv.fit(src_embeddings, tgt_embeddings)          # compute vector
        projection = cv.project(new_embeddings)         # scalar projection(s)
        sims = cv.cosine_similarity(new_embeddings)     # cosine similarity(s)

    Parameters:
        normalize (bool): If True, the concept vector is L2-normalized.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.vector: Optional[np.ndarray] = None
        self.original_vector: Optional[np.ndarray] = None
        self.src_mean: Optional[np.ndarray] = None
        self.tgt_mean: Optional[np.ndarray] = None
        self.dim: Optional[int] = None

    def _to_2d(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Embeddings must be 1D or 2D array-like")
        return arr
    def normalize_vector(self):
        """L2-normalize the concept vector in place."""
        if self.vector is None:
            raise RuntimeError("Concept vector not set. Call fit() first.")
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    def fit(self, src_embeddings, tgt_embeddings):
        """
        Compute the concept vector from source and target embeddings.

        src_embeddings: array-like, shape (n_src, d) or (d,)
        tgt_embeddings: array-like, shape (n_tgt, d) or (d,)
        """
        src = self._to_2d(src_embeddings)
        tgt = self._to_2d(tgt_embeddings)

        if src.shape[1] != tgt.shape[1]:
            raise ValueError("srcitive and tgtative embeddings must have same dimensionality")

        self.dim = src.shape[1]
        self.src_mean = src.mean(axis=0)
        self.tgt_mean = tgt.mean(axis=0)
        if self.src_mean is None or self.tgt_mean is None:
            raise RuntimeError("Source/target means not computed.")
        vec = self.tgt_mean - self.src_mean
        self.original_vector = vec
        self.vector = vec
        if self.normalize:
            self.normalize_vector()
        return self

    def project(self, embeddings):
        """
        Project embeddings onto the concept vector.

        Returns scalar projection(s). If embeddings is shape (n, d) returns (n,),
        if embeddings is shape (d,) returns a scalar.
        """
        if self.vector is None:
            raise RuntimeError("Concept vector not set. Call fit() first.")
        emb = self._to_2d(embeddings)
        if emb.shape[1] != self.dim:
            raise ValueError("Embedding dimensionality mismatch")
        # If vector is normalized, dot gives signed magnitude along the concept.
        scores = emb.dot(self.vector)
        return scores if scores.shape[0] > 1 else float(scores[0])

    def as_array(self):
        """Return the concept vector as a 1D numpy array (or None)."""
        return None if self.vector is None else np.asarray(self.vector)