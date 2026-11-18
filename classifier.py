"""Runtime hierarchical proof classifier (model loading and prediction only).

Overview
--------
This module is the *inference* side of a two-stage (hierarchical) text
classifier for mathematical proofs. Training happens elsewhere
(``train_classifier.py``); this file only loads the trained artifacts and
uses them to make predictions.

How a prediction works
----------------------
    question + proof
          |
          v
    build_text()                  -> one combined string
          |
          v
    MiniLMEncoder.encode()        -> one embedding vector (1 x D)
          |
          +--> subject_model            -> subject (e.g. "algebra")
          |
          +--> family_models[subject]   -> proof family (e.g. "induction")
                                           chosen *because of* the subject

The same embedding vector is fed to both stages. The second stage is picked
based on the first stage's answer, which is why the classifier is called
"hierarchical".

Expected model bundle
---------------------
The file passed as ``bundle_path`` is a ``joblib`` dump of a dict with:

    {
        "model_name":     str,   # (optional) name of the embedding model used
                                 #   in training; falls back to MODEL_NAME
        "subject_model":  Model, # classifier that predicts the subject
        "family_models":  {      # one classifier per subject
            "<subject>": Model,
            ...
        },
    }

where ``Model`` is either:

* a scikit-learn-style estimator exposing ``predict_proba`` and ``classes_``, or
* a "constant" placeholder dict ``{"kind": "constant", "label": "<label>"}``,
  used when training data contained only one class, so there was nothing to
  fit (it always predicts that label with probability 1.0).

Example
-------
>>> clf = HierarchicalProofClassifier("models/bundle.joblib")
>>> result = clf.predict("Show that n^2 + n is even.", "By induction on n ...")
>>> result["subject"], result["proof_family"]
('number_theory', 'induction')
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from embedding import MiniLMEncoder, MODEL_NAME
from parameter_schema import build_text


class HierarchicalProofClassifier:
    """Predict a subject, then a proof family conditional on that subject.

    Stage 1 predicts the mathematical *subject* of the input. Stage 2 uses a
    separate classifier that was trained only on proofs of that subject to
    predict the *proof family* (the general technique used).

    Attributes:
        bundle_path (Path): Location of the loaded model bundle on disk.
        bundle (dict): The deserialized bundle (see module docstring for the
            expected keys).
        encoder (MiniLMEncoder): Sentence-embedding model that turns text
            into the numeric vector the classifiers expect.
    """

    def __init__(self, bundle_path: str | Path, device: str = "cpu") -> None:
        """Load the trained bundle and prepare the text encoder.

        Args:
            bundle_path: Path to the ``joblib`` file produced by
                ``train_classifier.py``.
            device: Device for the embedding model, e.g. ``"cpu"`` or
                ``"cuda"``. Defaults to ``"cpu"``.

        Raises:
            FileNotFoundError: If ``bundle_path`` does not exist. The message
                tells the user to run the training script first.
        """
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {self.bundle_path}. "
                "Train it first with `python train_classifier.py`."
            )
        self.bundle = joblib.load(self.bundle_path)
        # Use the same embedding model that was used during training, so that
        # vectors at inference time live in the same space as training vectors.
        # Older bundles may lack "model_name", hence the MODEL_NAME fallback.
        self.encoder = MiniLMEncoder(
            model_name=self.bundle.get("model_name", MODEL_NAME), device=device
        )

    @staticmethod
    def _probability_dict(model: Any, vector: np.ndarray) -> dict[str, float]:
        """Turn any supported model into a ``{label: probability}`` mapping.

        Handles the two kinds of model that can appear in the bundle:

        * **Constant model** (``{"kind": "constant", "label": ...}``): returns
          that single label with probability ``1.0`` and never touches the
          vector.
        * **Regular estimator**: calls ``predict_proba`` on the vector and
          pairs each probability with the matching entry of ``model.classes_``.

        Args:
            model: A constant-model dict or an estimator with
                ``predict_proba`` and ``classes_``.
            vector: Embedding of shape ``(1, D)`` (a single sample).

        Returns:
            Dict mapping each label (as ``str``) to its probability (as
            ``float``). Probabilities of a regular estimator sum to ~1.
        """
        if isinstance(model, dict) and model.get("kind") == "constant":
            return {str(model["label"]): 1.0}
        # predict_proba returns shape (n_samples, n_classes); we only have one
        # sample, so take row 0.
        probabilities = model.predict_proba(vector)[0]
        return {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

    @staticmethod
    def _predict_from_model(model: Any, vector: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Run one classification stage and pick its top label.

        Args:
            model: A constant-model dict or an estimator (see
                :meth:`_probability_dict`).
            vector: Embedding of shape ``(1, D)``.

        Returns:
            A tuple ``(label, score, distribution)`` where:

            * ``label``: the most probable class,
            * ``score``: its probability (a confidence value in [0, 1]),
            * ``distribution``: the full ``{label: probability}`` mapping.
        """
        distribution = HierarchicalProofClassifier._probability_dict(model, vector)
        # max() over the keys, using each key's probability as the sort value.
        label = max(distribution, key=distribution.get)
        return label, distribution[label], distribution

    def predict(self, question: str, proof: str) -> dict[str, Any]:
        """Classify a question/proof pair into a subject and a proof family.

        Steps:
            1. Combine ``question`` and ``proof`` into one string with
               ``build_text``.
            2. Encode that string into an embedding vector.
            3. Predict the subject with ``bundle["subject_model"]``.
            4. Look up ``bundle["family_models"][subject]`` and use it to
               predict the proof family from the same vector.

        Args:
            question: The problem statement (may be empty if ``proof`` is
                given).
            proof: The proof text (may be empty if ``question`` is given).

        Returns:
            A dict with these keys:

            * ``"subject"`` (str): predicted subject.
            * ``"subject_score"`` (float): probability of that subject.
            * ``"subject_probabilities"`` (dict[str, float]): all subjects,
              sorted from most to least likely.
            * ``"proof_family"`` (str): predicted proof family, *given* the
              predicted subject.
            * ``"family_score"`` (float): probability of that family. Note
              this is conditional on the subject being correct.
            * ``"family_probabilities"`` (dict[str, float]): all families for
              that subject, sorted from most to least likely.

        Raises:
            ValueError: If ``build_text`` returns an empty string, i.e. both
                inputs were empty.
            KeyError: If the predicted subject has no entry in
                ``bundle["family_models"]`` (indicates an inconsistent
                bundle).
        """
        text = build_text(question, proof)
        if not text:
            raise ValueError("Please provide at least a question or a proof.")
        # show_progress_bar=False keeps the console clean for a single input.
        vector = self.encoder.encode([text], show_progress_bar=False)

        # Stage 1: which subject is this?
        subject, subject_score, subject_probs = self._predict_from_model(
            self.bundle["subject_model"], vector
        )
        # Stage 2: given that subject, which proof family? Each subject has
        # its own specialised classifier.
        family_model = self.bundle["family_models"][subject]
        family, family_score, family_probs = self._predict_from_model(
            family_model, vector
        )
        return {
            "subject": subject,
            "subject_score": subject_score,
            "subject_probabilities": dict(sorted(subject_probs.items(), key=lambda item: -item[1])),
            "proof_family": family,
            "family_score": family_score,
            "family_probabilities": dict(sorted(family_probs.items(), key=lambda item: -item[1])),
        }