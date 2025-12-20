"""
nanoTransE: minimal TransE for Knowledge Graph embeddings.
Translation-based scoring f(h,t) = ||h + r - t||_2. Torch only.
"""

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 1. Data: toy financial triples (Head, Relation, Tail) -> integer IDs
# -----------------------------------------------------------------------------
TRIPLES_STR = [
    ("Account_A", "Transferred_To", "Account_B"),
    ("Account_B", "Transferred_To", "Account_C"),
    ("Account_C", "Transferred_To", "Account_D"),
    ("Account_A", "Owned_By", "Customer_1"),
    ("Account_B", "Owned_By", "Customer_1"),
    ("Account_C", "Owned_By", "Customer_2"),
    ("Account_D", "Owned_By", "Customer_2"),
    ("Customer_1", "Linked_To", "Customer_2"),
]
entities = sorted(set(t[0] for t in TRIPLES_STR) | set(t[2] for t in TRIPLES_STR))
relations = sorted(set(t[1] for t in TRIPLES_STR))
entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}
n_entities, n_relations = len(entity2id), len(relation2id)
triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in TRIPLES_STR]


# -----------------------------------------------------------------------------
# 2. TransE: embeddings for entities and relations; score = ||h + r - t||_2
# -----------------------------------------------------------------------------
class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=32):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.entity_emb.weight, -0.01, 0.01)
        nn.init.uniform_(self.relation_emb.weight, -0.01, 0.01)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return (h + r - t).pow(2).sum(dim=1).sqrt()  # L2 distance


# -----------------------------------------------------------------------------
# 3. Corrupt triple: replace head or tail with random entity (negative sample)
# -----------------------------------------------------------------------------
def corrupt_triple(h, r, t, n_entities):
    if torch.rand(1).item() < 0.5:
        h_neg = torch.randint(0, n_entities, (h.shape[0],), device=h.device)
        return h_neg, r, t
    else:
        t_neg = torch.randint(0, n_entities, (t.shape[0],), device=t.device)
        return h, r, t_neg


# -----------------------------------------------------------------------------
# 4. Margin ranking loss: L = max(0, γ + f(h,r,t) - f(h',r,t'))
# -----------------------------------------------------------------------------
def margin_loss(pos_scores, neg_scores, margin=1.0):
    return (pos_scores - neg_scores + margin).clamp(min=0).mean()


# -----------------------------------------------------------------------------
# 5. Training loop
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransE(n_entities, n_relations, dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs, batch_size = 200, 4

    for epoch in range(epochs):
        perm = torch.randperm(len(triples))
        total_loss = 0.0
        for start in range(0, len(triples), batch_size):
            idx = perm[start : start + batch_size]
            batch = [triples[i] for i in idx]
            h = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
            r = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
            t = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
            h_neg, r_neg, t_neg = corrupt_triple(h, r, t, n_entities)

            pos_scores = model(h, r, t)
            neg_scores = model(h_neg, r_neg, t_neg)
            loss = margin_loss(pos_scores, neg_scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}, loss = {total_loss / (len(triples) // batch_size + 1):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
