"""Frozen MiniLM sentence embedding utilities shared by training and inference.

Overview
--------
Classifiers cannot read raw text, so this module converts text into numeric
vectors ("embeddings") using a pretrained sentence-embedding model,
``all-MiniLM-L6-v2``. Each piece of text becomes one fixed-length vector
(384 numbers for this model) that captures its meaning; texts with similar
meaning get vectors that point in similar directions.

"Frozen" means the embedding model is used exactly as downloaded and is never
fine-tuned. Only the small classifiers built on top of the embeddings are
trained.

Because training and inference must use the *same* embedding model (otherwise
the vectors would live in different spaces), both sides import from this file:

* Training  -> :func:`make_embeddings` (many texts at once, with a progress bar)
* Inference -> :class:`MiniLMEncoder` (used by ``HierarchicalProofClassifier``)

Contents
--------
* ``MODEL_NAME``      - default Hugging Face model identifier.
* ``MiniLMEncoder``   - lazy-loading wrapper around the embedding model.
* ``make_embeddings`` - one-call helper for training scripts.

Example
-------
>>> encoder = MiniLMEncoder()
>>> vectors = encoder.encode(["Prove by induction.", "Use a contradiction."])
>>> vectors.shape
(2, 384)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Default embedding model (Hugging Face identifier). It is small and fast and
#: produces 384-dimensional vectors.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEncoder:
    """Turn text into normalized embedding vectors using MiniLM.

    The underlying ``SentenceTransformer`` model is loaded *lazily*: creating
    an encoder is instant and cheap, and the (slow, memory-heavy) model load
    only happens the first time :meth:`encode` is called. The model is then
    cached on the instance and reused.

    Attributes:
        model_name (str): Hugging Face identifier of the embedding model.
        # device (str): Device the model runs on, e.g. ``"cpu"`` or ``"cuda"``.
        _model: The loaded ``SentenceTransformer``, or ``None`` until first use.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu") -> None:
        """Configure the encoder without loading the model yet.

        Args:
            model_name: Hugging Face model identifier. Defaults to
                ``MODEL_NAME``. Must match the model used at training time.
            device: Where to run the model (``"cpu"``, ``"cuda"``, ...).
        """
        self.model_name = model_name
        self.device = device
        # Not loaded yet; see _load().
        self._model = None

    def _load(self):
        """Load the embedding model on first use and return it.

        Subsequent calls return the already-loaded model instead of loading
        again. ``sentence_transformers`` is imported here (not at the top of
        the file) so that importing this module stays fast and does not fail
        for code that never actually encodes text.

        Returns:
            The cached ``SentenceTransformer`` instance.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed. The
                message tells the user to run
                ``pip install -r requirements.txt``.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name
            )  #   , device=self.device)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Convert a list of texts into a matrix of embedding vectors.

        Args:
            texts: The texts to encode (a non-empty list, tuple, etc.). Even
                a single text must be wrapped in a list, e.g. ``[text]``.
            batch_size: How many texts the model processes at once. Larger
                values are faster but use more memory.
            show_progress_bar: If ``True``, display a progress bar while
                encoding (useful for large training sets).

        Returns:
            A ``float32`` NumPy array of shape ``(len(texts), D)``, where
            ``D`` is the model's embedding size (384 for MiniLM-L6). Row ``i``
            is the embedding of ``texts[i]``. Every row is L2-normalized
            (length 1), so the dot product of two rows equals their cosine
            similarity.

        Raises:
            ValueError: If ``texts`` is empty.
            RuntimeError: If ``sentence-transformers`` is not installed
                (raised from :meth:`_load`).
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")
        vectors = self._load().encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,  # return a NumPy array, not a torch tensor
            normalize_embeddings=True,  # scale each vector to unit length
        )
        # Force float32 to keep memory use and saved models consistent.
        return np.asarray(vectors, dtype=np.float32)


def make_embeddings(
    texts: Sequence[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Convenience function used by classifier training.

    Creates a throwaway :class:`MiniLMEncoder` on the CPU, encodes all
    ``texts`` with a progress bar, and returns the result. Handy for training
    scripts that embed the whole dataset once and do not need to keep the
    encoder around.

    Args:
        texts: Non-empty sequence of texts to embed.
        model_name: Hugging Face model identifier. Defaults to ``MODEL_NAME``.
        batch_size: Number of texts processed per batch.

    Returns:
        ``float32`` array of shape ``(len(texts), D)`` with L2-normalized rows
        (see :meth:`MiniLMEncoder.encode`).

    Raises:
        ValueError: If ``texts`` is empty.
        RuntimeError: If ``sentence-transformers`` is not installed.
    """
    return MiniLMEncoder().encode(texts, batch_size=batch_size, show_progress_bar=True)
