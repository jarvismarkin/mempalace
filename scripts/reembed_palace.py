"""
reembed_palace.py — Re-embed a palace's drawers with the model configured in
the "embedding" block of ~/.mempalace/config.json.

Copies every drawer (id, document, metadata) into a new cosine-space
collection embedded by the configured model, verifies the count, then swaps
it in under the original name. Back up the palace directory first.

    python scripts/reembed_palace.py /path/to/palace [--collection mempalace_drawers]
"""

import argparse
import sys
import time

import chromadb

from mempalace.embedding import get_embedding_function, embedding_config

PAGE = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("palace")
    ap.add_argument("--collection", default="mempalace_drawers")
    args = ap.parse_args()

    ef = get_embedding_function()
    if ef is None:
        sys.exit("No ollama embedding configured in ~/.mempalace/config.json")
    print(f"Model: {embedding_config()['model']}  Palace: {args.palace}")

    client = chromadb.PersistentClient(path=args.palace)
    old = client.get_collection(args.collection)
    total = old.count()
    tmp_name = args.collection + "_reembed"
    try:
        client.delete_collection(tmp_name)
    except Exception:
        pass
    new = client.create_collection(
        tmp_name, metadata={"hnsw:space": "cosine"}, embedding_function=ef
    )

    start = time.time()
    for offset in range(0, total, PAGE):
        page = old.get(offset=offset, limit=PAGE, include=["documents", "metadatas"])
        new.add(ids=page["ids"], documents=page["documents"], metadatas=page["metadatas"])
        done = min(offset + PAGE, total)
        print(f"  {done}/{total}  ({time.time() - start:.0f}s)", flush=True)

    if new.count() != total:
        sys.exit(f"Count mismatch: {new.count()} != {total}; old collection left untouched")

    client.delete_collection(args.collection)
    new.modify(name=args.collection)
    print(f"Done: {total} drawers re-embedded in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
