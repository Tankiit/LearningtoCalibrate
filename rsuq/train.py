"""rsuq.train — Train the mass head (cluster-level next-token CE).

Self-contained (no dependency on the separate stage2 package). The mass head
is the only trained component anywhere in the program; the backbone is frozen.
Supervision: CE(head(h_t), kappa(gold_next_token)) — the disjoint-frame
special case of RS-NN belief encoding (one-hot on the host cluster). Softmax
over mass logits keeps m on the simplex (no Mobius, no clipping).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_mass_head(head, h_states, gold_ids, frame, device="cuda",
                    epochs=3, batch_size=1024, lr=1e-3, val_frac=0.1, seed=0):
    """h_states: (N, d) cached hidden states; gold_ids: (N,) next-token ids."""
    torch.manual_seed(seed)
    targets = frame.kappa[gold_ids]                       # (N,) cluster ids
    N = h_states.shape[0]
    perm = torch.randperm(N)
    n_val = int(N * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    dl = DataLoader(TensorDataset(h_states[tr], targets[tr]),
                    batch_size=batch_size, shuffle=True)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for ep in range(epochs):
        head.train()
        tot = nb = 0
        for hb, yb in dl:
            loss = F.cross_entropy(head(hb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep}: train CE {tot/nb:.4f}")
    head.eval()
    with torch.no_grad():
        logits = head(h_states[va].to(device))
        val_ce = F.cross_entropy(logits, targets[va].to(device)).item()
        val_acc = (logits.argmax(-1).cpu() == targets[va]).float().mean().item()
    print(f"  val: CE {val_ce:.4f}  cluster-acc {val_acc:.4f}")
    return {"val_ce": val_ce, "val_cluster_acc": val_acc}
