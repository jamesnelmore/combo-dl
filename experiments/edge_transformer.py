"""Autoregressive edge transformer for graph inequality search.

Generates graphs as variable-length sequences of edge tokens using next-token
prediction. Trains via a DCE-style loop: sample graphs, score with an inequality,
train on elites.

No positional embedding — edge order is meaningless for graphs.
Causal attention mask by default, configurable via constructor flag.
"""

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from combo_dl import WagnerCorollary21
from combo_dl.graph_utils import edge_vec_to_adj


# --- Model ---


class EdgeTransformer(nn.Module):
    """Autoregressive transformer that generates graphs as edge-token sequences.

    Vocabulary: edge tokens [0, num_edges), PAD, EOS.
    """

    def __init__(
        self,
        n: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.n = n
        self.num_edges = (n * (n - 1)) // 2
        self.causal = causal

        # Special tokens
        self.pad_token = self.num_edges
        self.eos_token = self.num_edges + 1
        self.vocab_size = self.num_edges + 2

        self.d_model = d_model

        self.token_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.output_head = nn.Linear(d_model, self.vocab_size)

    def forward(
        self, token_ids: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning next-token logits at every position.

        Args:
            token_ids: (batch, seq_len) token IDs.
            padding_mask: (batch, seq_len) True where padded.

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        seq_len = token_ids.shape[1]
        x = self.token_embedding(token_ids)

        causal_mask = None
        if self.causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=token_ids.device
            )
            # Match types to avoid PyTorch deprecation warning
            padding_mask = padding_mask.float()
            padding_mask = padding_mask.masked_fill(padding_mask == 1.0, float("-inf"))

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output_head(x)

    def sample(self, batch_size: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressively sample graphs, returned as binary edge vectors.

        Args:
            batch_size: Number of graphs to sample.
            temperature: Sampling temperature.

        Returns:
            (batch_size, num_edges) binary edge vectors.
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Track sequences and which samples are still generating
                sequences: list[list[int]] = [[] for _ in range(batch_size)]
                active = torch.ones(batch_size, dtype=torch.bool, device=device)
                # Track chosen edges per sample to prevent duplicates
                chosen = torch.zeros(batch_size, self.num_edges, dtype=torch.bool, device=device)

                for _ in range(self.num_edges):
                    if not active.any():
                        break

                    # Build padded input from current sequences
                    token_ids, padding_mask = self._build_sample_input(sequences, device)

                    logits = self.forward(token_ids, padding_mask)

                    # Get logits at the last real position for each sample
                    last_logits = self._get_last_position_logits(
                        logits, sequences, device
                    )

                    # Mask already-chosen edges
                    last_logits[:, :self.num_edges][chosen] = float("-inf")
                    # Never predict PAD
                    last_logits[:, self.pad_token] = float("-inf")

                    # Sample
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)

                    # Update sequences and state
                    for i in range(batch_size):
                        if not active[i]:
                            continue
                        tok = sampled[i].item()
                        if tok == self.eos_token:
                            active[i] = False
                        else:
                            sequences[i].append(tok)
                            chosen[i, tok] = True
        finally:
            if was_training:
                self.train()

        # Convert token sequences to binary edge vectors
        edge_vectors = torch.zeros(batch_size, self.num_edges, device=device)
        for i, seq in enumerate(sequences):
            if seq:
                edge_vectors[i, seq] = 1.0
        return edge_vectors

    def _build_sample_input(
        self, sequences: list[list[int]], device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded token_ids and padding_mask from current sequences.

        Empty sequences get a single EOS token as a start-of-sequence prompt.
        """
        batch_size = len(sequences)
        lengths = [max(len(s), 1) for s in sequences]
        max_len = max(lengths)

        token_ids = torch.full(
            (batch_size, max_len), self.pad_token, dtype=torch.long, device=device
        )
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        for i, seq in enumerate(sequences):
            if seq:
                length = len(seq)
                token_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
                padding_mask[i, :length] = False
            else:
                # Empty sequence: use EOS as start token
                token_ids[i, 0] = self.eos_token
                padding_mask[i, 0] = False

        return token_ids, padding_mask

    def _get_last_position_logits(
        self,
        logits: torch.Tensor,
        sequences: list[list[int]],
        device: torch.device | str,
    ) -> torch.Tensor:
        """Extract logits at the last non-padded position for each sample."""
        batch_size = logits.shape[0]
        last_indices = torch.tensor(
            [max(len(s) - 1, 0) for s in sequences], dtype=torch.long, device=device
        )
        return logits[torch.arange(batch_size, device=device), last_indices]


# --- Utilities ---


def edge_vec_to_token_seq(edge_vec: torch.Tensor) -> list[int]:
    """Convert a single binary edge vector to a sorted list of edge token IDs."""
    return edge_vec.nonzero(as_tuple=True)[0].tolist()


# --- Training loop ---


def train(
    n: int = 18,
    max_iterations: int = 10_000,
    batch_size: int = 64,
    elite_proportion: float = 0.1,
    lr: float = 1e-3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 64,
    early_stopping_patience: int = 300,
    seed: int = 42,
):
    torch.manual_seed(seed)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    problem = WagnerCorollary21(n=n)

    model = EdgeTransformer(
        n=n,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Graph size: n={n}, possible edges={model.num_edges}")
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token)

    best_score = float("-inf")
    best_construction = None
    steps_since_best = 0

    pbar = tqdm(range(max_iterations), desc="Iterations")
    for iteration in pbar:
        # 1. Sample
        model.eval()
        edge_vectors = model.sample(batch_size)

        # 2. Score (penalize disconnected graphs)
        scores = problem.reward(edge_vectors)
        adj_batch = edge_vec_to_adj(edge_vectors, n)
        for i in range(batch_size):
            G = nx.from_numpy_array(adj_batch[i].cpu().numpy())
            if not nx.is_connected(G):
                scores[i] = float("-inf")
        avg_score = scores[scores > float("-inf")].mean().item() if (scores > float("-inf")).any() else float("-inf")
        current_best = scores.max().item()

        found_new_best = False
        if current_best > best_score:
            best_score = current_best
            best_idx = torch.argmax(scores)
            best_construction = edge_vectors[best_idx].clone()
            steps_since_best = 0
            found_new_best = True
        else:
            steps_since_best += 1

        # 3. Check stopping
        should_stop, reason = problem.should_stop_early(best_score)
        if should_stop:
            tqdm.write(f"Stopping: {reason}")
            break

        if steps_since_best >= early_stopping_patience:
            tqdm.write(f"No improvement for {steps_since_best} iterations. Stopping.")
            break

        # 4. Select elites
        num_elites = max(1, int(batch_size * elite_proportion))
        elite_indices = torch.argsort(scores, descending=True)[:num_elites]
        elite_vectors = edge_vectors[elite_indices]

        # 5. Train on elites via next-token prediction
        model.train()

        # Convert elite edge vectors to token sequences (sorted edge IDs + EOS)
        elite_seqs = []
        for vec in elite_vectors:
            seq = edge_vec_to_token_seq(vec)
            seq.append(model.eos_token)
            elite_seqs.append(seq)

        # Build teacher-forcing batch: input = seq[:-1], target = seq[1:]
        # For sequences like [e3, e7, EOS]: input [e3, e7], target [e7, EOS]
        # But the first token also needs a target, so:
        # input = full seq, target = shifted seq
        max_seq_len = max(len(s) for s in elite_seqs)

        input_ids = torch.full(
            (num_elites, max_seq_len), model.pad_token, dtype=torch.long, device=device
        )
        target_ids = torch.full(
            (num_elites, max_seq_len), model.pad_token, dtype=torch.long, device=device
        )
        input_mask = torch.ones(
            num_elites, max_seq_len, dtype=torch.bool, device=device
        )

        for i, seq in enumerate(elite_seqs):
            # Input: all tokens except the last (EOS)
            input_len = len(seq) - 1
            if input_len > 0:
                input_ids[i, :input_len] = torch.tensor(
                    seq[:-1], dtype=torch.long
                )
                target_ids[i, :input_len] = torch.tensor(
                    seq[1:], dtype=torch.long
                )
                input_mask[i, :input_len] = False
            else:
                # Empty graph: just predict EOS from start token
                input_ids[i, 0] = model.eos_token
                target_ids[i, 0] = model.eos_token
                input_mask[i, 0] = False

        logits = model(input_ids, input_mask)  # (elites, seq_len, vocab_size)

        loss = criterion(
            logits.reshape(-1, model.vocab_size),
            target_ids.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Progress
        pbar.set_postfix({
            "best": f"{best_score:.4f}",
            "avg": f"{avg_score:.4f}",
            "loss": f"{loss.item():.4f}",
            "since_best": steps_since_best,
        })
        if found_new_best:
            tqdm.write(
                f"Iter {iteration}: new best score {best_score:.4f} "
                f"(goal: {problem.goal_score:.4f})"
            )

    # Print best graph
    if best_construction is not None:
        print(f"\nBest score: {best_score:.4f} (goal: {problem.goal_score:.4f})")
        adj = edge_vec_to_adj(best_construction.unsqueeze(0), n).squeeze(0)
        G = nx.from_numpy_array(adj.cpu().numpy())
        print(f"Vertices: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"Degree sequence: {sorted((d for _, d in G.degree()), reverse=True)}")
        print(f"Edge list: {sorted(G.edges())}")


if __name__ == "__main__":
    train()
