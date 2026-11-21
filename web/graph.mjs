// A lightweight, source-backed concept graph. Edges mean sentence co-occurrence,
// not a verified semantic relation. No API call or database service is needed.
const STOP = new Set(('the a an and or of to in on for from with by is are was were be been this that these those it its as at not can will has have had each all also into using use used how what which when where why who does do your you their they then than only any more such may should through over under between about our we if else').split(' '));
export const terms = text => (text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}_-]*/gu) || []).filter(t => t.length > 2 && !STOP.has(t) && !/^\d+$/.test(t));
export function buildGraph(chunks) {
  const counts = new Map();
  for (const c of chunks) for (const t of new Set(terms(c.text))) counts.set(t, (counts.get(t)||0)+1);
  const vocab = new Set([...counts].sort((a,b)=>b[1]-a[1]).slice(0,180).map(([t])=>t));
  const nodes = new Map(), edges = new Map();
  chunks.forEach((c,index) => {
    for (const sentence of c.text.split(/[.!?\n]+/)) {
      const ts = [...new Set(terms(sentence))].filter(t=>vocab.has(t)).slice(0,18);
      for (const t of ts) { if (!nodes.has(t)) nodes.set(t,{id:t,chunks:new Set(),neighbors:new Map()}); nodes.get(t).chunks.add(index); }
      for (let i=0;i<ts.length;i++) for (let j=i+1;j<ts.length;j++) {
        const [a,b]=[ts[i],ts[j]].sort(), key=JSON.stringify([a,b]);
        if (!edges.has(key)) edges.set(key,{a,b,chunks:new Set()});
        edges.get(key).chunks.add(index);
        nodes.get(a).neighbors.set(b,(nodes.get(a).neighbors.get(b)||0)+1);
        nodes.get(b).neighbors.set(a,(nodes.get(b).neighbors.get(a)||0)+1);
      }
    }
  });
  return {nodes,edges};
}
export function searchGraph(graph,query,count) {
  const seeds=[...new Set(terms(query))].filter(t=>graph.nodes.has(t));
  const activation=new Map(seeds.map(t=>[t,1]));
  let frontier=seeds.map(t=>[t,1]);
  for(let hop=0;hop<2;hop++) {
    const next=[];
    for(const [id,weight] of frontier) {
      const ns=[...graph.nodes.get(id).neighbors].sort((a,b)=>b[1]-a[1]).slice(0,8);
      const total=ns.reduce((s,[,w])=>s+w,0)||1;
      for(const [n,w] of ns) {const value=weight*0.5*w/total;activation.set(n,(activation.get(n)||0)+value);next.push([n,value]);}
    }
    frontier=next;
  }
  const scores=Array(count).fill(0);
  for(const [id,weight] of activation) for(const i of graph.nodes.get(id).chunks) scores[i]+=weight/Math.sqrt(graph.nodes.get(id).chunks.size);
  const paths=[...graph.edges.values()].filter(e=>seeds.includes(e.a)||seeds.includes(e.b)).sort((a,b)=>b.chunks.size-a.chunks.size).slice(0,8);
  return {scores,seeds,paths,active:activation};
}
