"""Free local concept graph. Links mean sentence co-occurrence, not facts.

Uses the same retrieval idea as web/graph.mjs, with source chunk IDs on edges.
No Neo4j service, paid extraction API, or extra dependency is required.
"""
import re
from collections import Counter, defaultdict

STOP = set('the a an and or of to in on for from with by is are was were be been this that these those it its as at not can will has have had each all also into using use used how what which when where why who does do your you their they then than only any more such may should through over under between about our we if else'.split())


def terms(text):
    return [t for t in re.findall(r'\w[\w-]*', text.lower()) if len(t)>2 and t not in STOP and not t.isdigit()]


def build_graph(texts):
    counts = Counter(t for text in texts for t in set(terms(text)))
    vocab = {t for t, _ in counts.most_common(180)}
    nodes, neighbors, edges = defaultdict(set), defaultdict(Counter), defaultdict(set)
    for index, text in enumerate(texts):
        for sentence in re.split(r'[.!?\n]+', text):
            ts = list(dict.fromkeys(t for t in terms(sentence) if t in vocab))[:18]
            for t in ts:
                nodes[t].add(index)
            for i, a in enumerate(ts):
                for b in ts[i+1:]:
                    neighbors[a][b] += 1
                    neighbors[b][a] += 1
                    edges[tuple(sorted((a,b)))].add(index)
    return dict(nodes=nodes, neighbors=neighbors, edges=edges, count=len(texts))


def graph_search(graph, question):
    seeds = set(terms(question)) & graph['nodes'].keys()
    active = Counter({s: 1.0 for s in seeds})
    frontier = dict(active)
    for _ in range(2):
        next_hop = Counter()
        for node, weight in frontier.items():
            ns = graph['neighbors'][node].most_common(8)
            total = sum(w for _,w in ns) or 1
            for n, w in ns:
                value = weight * .5 * w / total
                active[n] += value
                next_hop[n] += value
        frontier = next_hop
    scores = [0.] * graph['count']
    for node, weight in active.items():
        ids = graph['nodes'][node]
        for i in ids:
            scores[i] += weight / len(ids)**.5
    links = [(a,b,sorted(ids)) for (a,b),ids in graph['edges'].items() if a in seeds or b in seeds]
    return scores, sorted(links, key=lambda x:len(x[2]), reverse=True)[:8]


def combine_results(vector_docs, chunks, question, top_k=4):
    """Fuse the existing BM25+vector lane with nonzero graph results via RRF."""
    graph = build_graph([d.page_content for d in chunks])
    scores, links = graph_search(graph, question)
    order = sorted((i for i,s in enumerate(scores) if s>0), key=lambda i:scores[i], reverse=True)
    key = lambda d: (d.page_content, str(sorted(d.metadata.items())))
    docs = {key(d): d for d in vector_docs}
    fused = {key(d): .7/(60+rank) for rank,d in enumerate(vector_docs,1)}
    for rank, i in enumerate(order,1):
        d = chunks[i]
        docs[key(d)] = d
        fused[key(d)] = fused.get(key(d),0) + .3/(60+rank)
    return [docs[k] for k in sorted(fused,key=fused.get,reverse=True)[:top_k]], links
