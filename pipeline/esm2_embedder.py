import torch
import numpy as np

try:
    import esm  # provided by fair-esm / facebookresearch/esm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'esm' package is required for ESM2Embedder. "
        "Install it with 'pip install fair-esm'."
    ) from e


class ESM2Embedder:
    """
    Simple wrapper around Facebook's ESM2 model to obtain per-sequence embeddings.

    This implementation uses the CLS token (per-sequence) representation and returns
    a NumPy array of shape (N, D) for N sequences.
    """

    def __init__(self, model_name: str = "esm2_t12_35M_UR50D", device: torch.device | str = "cpu", batch_size: int = 1):
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Load pretrained model and Alphabet/batch converter.
        self.model, self.alphabet = esm.pretrained.__dict__[model_name]()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        # Use model's last layer dynamically (e.g. 12, 33, etc.).
        self.repr_layer = int(getattr(self.model, "num_layers", 12))

    @torch.no_grad()
    def embed(self, sequences: list[str]) -> np.ndarray:
        """
        Compute CLS embeddings for a list of amino-acid sequences.

        Returns
        -------
        np.ndarray
            Array of shape (N, D) with one embedding per input sequence.
        """
        all_embeddings: list[np.ndarray] = []

        # Process in batches for large lists
        for start in range(0, len(sequences), self.batch_size):
            batch_seqs = sequences[start : start + self.batch_size]
            batch_data = [(str(i), s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = self.batch_converter(batch_data)
            tokens = tokens.to(self.device)

            # Extract CLS representations (index 0) from selected repr layer
            out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            reps = out.get("representations", {})
            if self.repr_layer in reps:
                token_representations = reps[self.repr_layer]
            elif str(self.repr_layer) in reps:
                token_representations = reps[str(self.repr_layer)]
            elif reps:
                # Fallback: pick highest available layer key
                key = max(reps.keys(), key=lambda k: int(k))
                token_representations = reps[key]
            else:
                raise RuntimeError("ESM output does not include representations")
            cls_reprs = token_representations[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_reprs)

        return np.concatenate(all_embeddings, axis=0)

