"""
embedding.py — Pluggable embedding backend for the palace.

Without an "embedding" block in ~/.mempalace/config.json, collections use
ChromaDB's built-in default (all-MiniLM-L6-v2). With one, documents and
queries are embedded through a local Ollama model, using separate document
and query prefixes for asymmetric retrieval models such as EmbeddingGemma:

    "embedding": {
      "provider": "ollama",
      "model": "embeddinggemma",
      "url": "http://127.0.0.1:11434",
      "document_prefix": "title: none | text: ",
      "query_prefix": "task: search result | query: "
    }

Vectors from different models cannot be mixed — changing the model means
re-embedding the palace (see scripts/reembed_palace.py).
"""

import json
import os
import urllib.request

_CONFIG_FILE = os.path.expanduser("~/.mempalace/config.json")
_BATCH = 32
_ef_cache = {}


def embedding_config() -> dict:
    """Return the "embedding" block from config.json, or {} for the Chroma default."""
    try:
        with open(_CONFIG_FILE) as f:
            return json.load(f).get("embedding") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _ollama_embed(cfg: dict, texts: list) -> list:
    """Embed a list of inputs: strings, or {"image": b64} / {"audio": b64} items."""
    url = cfg.get("url", "http://127.0.0.1:11434").rstrip("/") + "/api/embed"
    out = []
    for i in range(0, len(texts), _BATCH):
        body = {"model": cfg["model"], "input": texts[i : i + _BATCH], "truncate": True}
        if cfg.get("dimensions"):
            body["dimensions"] = cfg["dimensions"]
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            out.extend(json.load(resp)["embeddings"])
    return out


class OllamaDocumentEmbedding:
    """Chroma embedding function: embeds documents with the document prefix."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(self, input):
        prefix = self.cfg.get("document_prefix", "")
        return _ollama_embed(self.cfg, [prefix + t for t in input])

    def name(self):
        return "ollama:" + self.cfg["model"]


def get_embedding_function():
    """Embedding function to pass to Chroma, or None to use Chroma's default."""
    cfg = embedding_config()
    if cfg.get("provider") != "ollama":
        return None
    key = json.dumps(cfg, sort_keys=True)
    if key not in _ef_cache:
        _ef_cache[key] = OllamaDocumentEmbedding(cfg)
    return _ef_cache[key]


def query_args(texts: list) -> dict:
    """kwargs for collection.query(): query_texts, or prefixed query_embeddings."""
    cfg = embedding_config()
    if cfg.get("provider") != "ollama":
        return {"query_texts": texts}
    prefix = cfg.get("query_prefix", "")
    return {"query_embeddings": _ollama_embed(cfg, [prefix + t for t in texts])}


def embed_media(kind: str, data_b64: str) -> list:
    """Embed one image or audio clip ("image" | "audio", base64) into the text space.

    Needs a multimodal model (embeddinggemma-2:740m). Audio is read as 16 kHz
    WAV and only roughly the first 30 s count, so callers segment longer clips.
    """
    cfg = embedding_config()
    if cfg.get("provider") != "ollama":
        raise RuntimeError("Media embedding needs an ollama embedding configured")
    return _ollama_embed(cfg, [{kind: data_b64}])[0]


def open_collection(client, name: str, create: bool = False, metadata: dict = None):
    """get_collection / get_or_create_collection with the configured embedding.

    New collections default to cosine space, which is what callers assume
    when they report similarity as 1 - distance.
    """
    ef = get_embedding_function()
    kwargs = {"embedding_function": ef} if ef else {}
    if create:
        return client.get_or_create_collection(
            name, metadata=metadata or {"hnsw:space": "cosine"}, **kwargs
        )
    return client.get_collection(name, **kwargs)
