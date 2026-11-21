"""Run from the Graph RAG folder: python tests/test_graph.py
(also works under: python -m unittest discover -s tests -p 'test_graph.py')."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_rag import build_graph, graph_search, combine_results

class GraphTests(unittest.TestCase):
    def setUp(self):
        self.texts = ['Alice manages Orion. Orion depends on Vega.', 'Vega supplies telescopes.', 'Bananas are yellow.']
        self.graph = build_graph(self.texts)
    def test_seed_and_two_hops(self):
        scores, _ = graph_search(self.graph, 'Alice')
        self.assertGreater(scores[0],0)
        self.assertGreater(scores[1],0)
        self.assertEqual(scores[2],0)
    def test_edge_provenance(self):
        self.assertEqual(self.graph['edges'][('alice','orion')], {0})
    def test_no_match(self):
        scores, links = graph_search(self.graph, 'unmatched')
        self.assertEqual(sum(scores),0)
        self.assertEqual(links,[])
    def test_empty_graph(self):
        self.assertEqual(graph_search(build_graph([]),'Alice'),([],[]))
    def test_fusion_deduplicates(self):
        docs=[SimpleNamespace(page_content=t,metadata={'page':i}) for i,t in enumerate(self.texts)]
        result,links=combine_results(docs,docs,'Alice',4)
        self.assertEqual(len(result),3)
        self.assertTrue(links)
        self.assertEqual(result[0].page_content,self.texts[0])
    def test_no_graph_match_preserves_original(self):
        docs=[SimpleNamespace(page_content=t,metadata={'page':i}) for i,t in enumerate(self.texts)]
        result,_=combine_results(docs,docs,'unmatched',4)
        self.assertEqual(result,docs)

if __name__=='__main__': unittest.main()
