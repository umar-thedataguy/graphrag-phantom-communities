"""LLM entity extraction via any OpenAI-compatible endpoint (e.g. Ollama).

NOT RUN in the committed results -- this build was executed in a CPU-only
sandbox without a local model server. Run it yourself with:

    ollama pull llama3.2:3b
    python src/extract_llm.py --book austen-emma --chunks 50

and compare node/edge recall against extract_regex on the same chunks.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

PROMPT = """Extract named entities (people, places, organizations) and the
relationships between them from the text below. Respond ONLY with JSON:
{{"entities": ["..."], "relations": [["entity_a", "entity_b"], ...]}}
Text:
{chunk}
"""


def call_llm(chunk: str, model: str, base_url: str) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(chunk=chunk)}],
        "stream": False,
        "format": "json",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    text = data["message"]["content"]
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="austen-emma")
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--chunk-chars", type=int, default=1500)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args()

    import nltk
    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    from extract_regex import extract_graph

    text = gutenberg.raw(f"{args.book}.txt")
    chunks = [text[i:i + args.chunk_chars]
              for i in range(0, len(text), args.chunk_chars)][:args.chunks]
    sample_text = "\n".join(chunks)

    G_regex = extract_graph(sample_text, min_count=2, min_edge_weight=1)

    nodes, edges = set(), set()
    for i, ch in enumerate(chunks):
        try:
            out = call_llm(ch, args.model, args.base_url)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {i}: LLM call failed ({e}); skipping")
            continue
        ents = {e.strip() for e in out.get("entities", []) if isinstance(e, str)}
        nodes |= ents
        for pair in out.get("relations", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                edges.add(tuple(sorted(str(x).strip() for x in pair)))
        print(f"chunk {i}: {len(ents)} entities, {len(out.get('relations', []))} relations")

    rn = len(G_regex.nodes())
    re_ = len(G_regex.edges())
    node_overlap = len({n for n in nodes if n in G_regex}) / rn if rn else 0
    edge_overlap = len({e for e in edges
                        if G_regex.has_edge(*e)}) / re_ if re_ else 0
    print(json.dumps({
        "regex_nodes": rn, "regex_edges": re_,
        "llm_nodes": len(nodes), "llm_edges": len(edges),
        "llm_node_recall_vs_regex": round(node_overlap, 3),
        "llm_edge_recall_vs_regex": round(edge_overlap, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
