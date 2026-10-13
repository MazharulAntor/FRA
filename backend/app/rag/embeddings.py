"""Embedding model wrapper. Only this class knows about sentence-transformers."""

BATCH_SIZE = 64


class Embedder:
    """Turns text into normalised vectors. Swap this class to change the model."""

    def __init__(self, model_name: str) -> None:
        # Imported here so tests that use a fake embedder never load torch.
        from sentence_transformers import SentenceTransformer

        name = model_name.removeprefix("sentence-transformers/")
        self._model = SentenceTransformer(name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in batches of 64; vectors have length 1."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
