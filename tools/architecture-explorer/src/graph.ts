import type { Connection, Layer, Module } from './types.ts';

export const layers: {id: Layer; title: string; color: string; subtitle: string}[] = [
  {id:'frontend',title:'Frontend',color:'#8ea8ff',subtitle:'React · TanStack · browser state'},
  {id:'backend',title:'Backend',color:'#ff967f',subtitle:'FastAPI · services · model inference'},
  {id:'database',title:'Database',color:'#65d7b0',subtitle:'SQLAlchemy models · PostgreSQL'},
  {id:'external',title:'External services',color:'#d9a2f0',subtitle:'Conditional providers & host hooks'},
  {id:'offline',title:'Offline & tooling',color:'#e8c177',subtitle:'Training · artifacts · migrations · tests'},
];
export const colorFor = (layer: Layer) => layers.find(x => x.id === layer)!.color;
export const NODE_WIDTH = 246;
export const NODE_HEIGHT = 112;
export interface Point {x: number; y: number}
export interface GroupLayout {id: Layer; x: number; y: number; width: number; height: number}
export function matches(node: Module, query: string) {
  const haystack = [node.title,node.description,node.note,...node.technologies,...node.files,
    ...node.routes.map(r => r.route),...node.endpoints.map(e => e.method+' '+e.route),...node.tables.map(t => t.name)].join(' ').toLowerCase();
  return query.trim().toLowerCase().split(/\s+/).every(term => haystack.includes(term));
}
export function filterGraph(nodes: Module[], edges: Connection[], enabled: Layer[], query: string, status: string, relation: string, focus: string[]) {
  const visible = nodes.filter(n => enabled.includes(n.layer) && matches(n,query) &&
    (status === 'all' || (status === 'active' ? n.status !== 'inactive' : n.status === status)) &&
    (!focus.length || focus.includes(n.id)));
  const ids = new Set(visible.map(n => n.id));
  return {nodes: visible, edges: edges.filter(e => ids.has(e.source) && ids.has(e.target) &&
    (relation === 'all' || (relation === 'runtime' ? e.kind !== 'foreign-key' : relation === 'conditional' ? e.conditional : e.kind === 'foreign-key')))};
}
// Pack each visible group independently so focused paths remain legible.
export function layoutGraph(nodes: Module[]) {
  const positions: Record<string, Point> = {};
  const relative: Record<string, Point> = {};
  const groups: GroupLayout[] = [];
  let x = 0;
  for (const layer of layers) {
    const members = nodes.filter(n => n.layer === layer.id);
    if (!members.length) continue;
    const cols = members.length > 7 ? 2 : 1;
    const width = cols * (NODE_WIDTH+34)+30;
    const height = Math.ceil(members.length/cols) * (NODE_HEIGHT+46)+84;
    groups.push({id:layer.id,x,y:0,width,height});
    members.forEach((n,i) => {
      const p = {x:30+(i%cols)*(NODE_WIDTH+34),y:84+Math.floor(i/cols)*(NODE_HEIGHT+46)};
      relative[n.id] = p;
      positions[n.id] = {x:x+p.x,y:p.y};
    });
    x += width+86;
  }
  return {positions,relative,groups};
}
export const absoluteFile = (root: string, path: string) => root.replace(/\\/g,'/').replace(/\/$/,'')+'/'+path;
