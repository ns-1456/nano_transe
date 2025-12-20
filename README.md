# nanoTransE: Knowledge Graph Embeddings

A minimal **TransE** implementation for multi-relational knowledge graphs. Translation-based embeddings using only PyTorch.

## Translation-based embedding

For a triple (head, relation, tail), TransE assumes:

**h + r ≈ t**

Entities and relations are represented as vectors in the same space. The relation vector **r** “translates” the head to the tail. The scoring function is the L2 distance:

**f(h, t) = ||h + r − t||_2**

A **lower** score means a more plausible triple. This is used for **link prediction**: ranking candidate tails (or heads) for a given (h, r) or (r, t).

## Training

We use margin-based ranking: positive triples should have a lower score than corrupted (negative) triples. Negatives are created by replacing the head or tail with a random entity. Loss:

**L = max(0, γ + f(h,r,t) − f(h′,r,t′))**

## How to run

```bash
pip install -r requirements.txt
python nano_transe.py
```

The script trains on a small hardcoded set of financial triples (accounts, customers, relations) and prints the loss as it converges.
