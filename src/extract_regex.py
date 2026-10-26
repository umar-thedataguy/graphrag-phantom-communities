"""Regex entity extraction: capitalized-name entities, sentence co-occurrence edges.

This is the deterministic baseline extractor. It is crude but transparent
and 100% reproducible, which is what an audit baseline needs.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import networkx as nx

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# One or more Capitalized words in sequence (e.g. "Emma Woodhouse", "Captain Ahab")
ENTITY = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")

# words that start sentences / honorifics that regex mistakes for entities
STOP = {
    "The", "And", "But", "She", "He", "They", "You", "His", "Her", "Him",
    "That", "This", "These", "Those", "There", "Then", "When", "What",
    "Which", "While", "Who", "Whose", "How", "Why", "Where", "With",
    "Was", "Were", "Had", "Has", "Have", "Not", "Nor", "For", "From",
    "Yes", "Yet", "Now", "All", "Any", "Are", "Our", "Your", "Their",
    "Its", "Upon", "Into", "Over", "Under", "After", "Before", "Though",
    "Chapter", "Volume", "Book", "Mrs", "Mr", "Miss", "Sir", "Lady",
    "Madam", "Dear", "Well", "Oh", "Ah", "If", "In", "On", "At", "As",
    "It", "Is", "Be", "By", "Do", "No", "So", "To", "Of", "Or", "An",
    "Let", "Come", "Till", "Such", "Some", "Both", "Even", "Ever",
    "Never", "Nothing", "Every", "Here", "Thus", "Thou", "Thy", "Thee",
    "Hath", "Doth", "Enter", "Exit", "Exeunt", "Scene", "Act",
}


def _clean(name: str) -> str | None:
    words = [w for w in name.split() if w not in STOP]
    if not words:
        return None
    return " ".join(words)


def extract_graph(text: str, min_count: int = 3,
                  min_edge_weight: int = 2) -> nx.Graph:
    """Entity co-occurrence graph from raw text.

    Nodes: regex entities appearing >= min_count times.
    Edges: pairs co-occurring in >= min_edge_weight sentences.
    Returns the largest connected component.
    """
    sentences = SENT_SPLIT.split(text)
    entity_counts: Counter = Counter()
    sent_entities: list[set[str]] = []
    for s in sentences:
        ents = set()
        for m in ENTITY.finditer(s):
            e = _clean(m.group(1))
            if e:
                ents.add(e)
        sent_entities.append(ents)
        entity_counts.update(ents)

    keep = {e for e, c in entity_counts.items() if c >= min_count}
    edge_w: Counter = Counter()
    for ents in sent_entities:
        kept = sorted(ents & keep)
        for a, b in combinations(kept, 2):
            edge_w[(a, b)] += 1

    G = nx.Graph()
    G.add_nodes_from(sorted(keep))
    for (a, b), w in sorted(edge_w.items()):
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        return G
    giant = max(nx.connected_components(G), key=len)
    return G.subgraph(giant).copy()
