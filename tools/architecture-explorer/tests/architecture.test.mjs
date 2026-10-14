import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { absoluteFile, filterGraph, layoutGraph, layers, matches, NODE_WIDTH, NODE_HEIGHT } from '../src/graph.ts';
import { diagramSVG, mermaidDiagram } from '../src/export.ts';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
const snapshot=JSON.parse(readFileSync(new URL('../src/architecture.json',import.meta.url),'utf8'));
const ids=new Set(snapshot.nodes.map(n=>n.id));
test('all mapped files exist and every evidence anchor matches its stated line',()=>{
  for(const n of snapshot.nodes) for(const p of n.files) assert.ok(existsSync(resolve(root,p)),p);
  for(const item of [...snapshot.nodes,...snapshot.edges]) {
    assert.ok(item.evidence.length, item.id);
    for(const e of item.evidence) {
      const line=readFileSync(resolve(root,e.path),'utf8').replace(/^\uFEFF/,'').split(/\r?\n/)[e.line-1];
      assert.ok(line.includes(e.anchor),`${item.id} ${e.path}:${e.line}`);
      assert.equal(line.trim(),e.excerpt);
    }
  }
});
test('normalized source hashes match; inventory excludes secrets, binaries and dependency trees',()=>{
  for(const f of snapshot.files) {
    const source=readFileSync(resolve(root,f.path),'utf8').replace(/^\uFEFF/,'').replace(/\r\n?/g,'\n');
    assert.equal(createHash('sha256').update(source).digest('hex'),f.sha256,f.path);
    assert.doesNotMatch(f.path,/(?:\.env|node_modules|\.venv|\.pth$|\.pkl$|\.pt$|\.csv$|tools\/architecture-explorer)/);
  }
});
test('all graph endpoints are valid and every application route/API is represented',()=>{
  assert.equal(ids.size,snapshot.nodes.length);
  assert.equal(new Set(snapshot.edges.map(e=>e.id)).size,snapshot.edges.length);
  for(const e of snapshot.edges){assert.ok(ids.has(e.source),e.id);assert.ok(ids.has(e.target),e.id);}
  for(const r of snapshot.routes) assert.ok(snapshot.nodes.some(n=>n.routes.some(x=>x.file===r.file&&x.route===r.route)),r.route);
  for(const r of snapshot.endpoints) assert.ok(snapshot.nodes.some(n=>n.endpoints.some(x=>x.file===r.file&&x.route===r.route&&x.method===r.method)),r.route);
  assert.equal(snapshot.endpoints.find(e=>e.method==='POST'&&e.route==='/api/assistant/chat').auth,'required');
  assert.equal(snapshot.endpoints.find(e=>e.route==='/api/recommendations/{user_id}').auth,'optional');
});
test('foreign key target tables and columns really exist',()=>{
  for(const t of snapshot.tables) for(const fk of t.foreignKeys) {
    const [target,column]=fk.target.split('.');
    const table=snapshot.tables.find(t=>t.name===target);
    assert.ok(table, fk.target);assert.ok(table.columns.some(c=>c.name===column),fk.target);
  }
  assert.equal(snapshot.tables.length,13);
  assert.equal(snapshot.tables.find(t=>t.name==='recommendation_studies').foreignKeys[0].target,'users.id');
  assert.equal(snapshot.tables.find(t=>t.name==='assistant_feedback').foreignKeys.length,1);
});
test('combined layer, search, status, path and relationship filters never leave dangling edges',()=>{
  for(const enabled of [layers.map(l=>l.id),['backend'],[],['database','frontend']]) {
    for(const query of ['','jwt','/api/assistant/chat','no-such-module']) for(const relation of ['all','runtime','foreign-key','conditional']) {
      const result=filterGraph(snapshot.nodes,snapshot.edges,enabled,query,'all',relation,[]);
      const visibleIds=new Set(result.nodes.map(n=>n.id));
      for(const e of result.edges) assert.ok(visibleIds.has(e.source)&&visibleIds.has(e.target));
      if(relation==='runtime')assert.ok(result.edges.every(e=>e.kind!=='foreign-key'));
      if(relation==='foreign-key')assert.ok(result.edges.every(e=>e.kind==='foreign-key'));
    }
  }
  assert.ok(matches(snapshot.nodes.find(n=>n.id==='auth-ui'),'AuthContext.tsx'));
  assert.ok(matches(snapshot.nodes.find(n=>n.id==='assistant-api'),'/api/assistant/chat'));
  assert.equal(filterGraph(snapshot.nodes,snapshot.edges,layers.map(l=>l.id),'','inactive','all',[]).nodes.length,2);
});
test('guided steps reference real nodes; focused layouts remain inside groups',()=>{
  for(const w of snapshot.walkthroughs) {
    for(const s of w.steps){assert.ok(ids.has(s[0]));assert.ok(s[1].length>20);}
    const result=filterGraph(snapshot.nodes,snapshot.edges,layers.map(l=>l.id),'','all','all',w.steps.map(s=>s[0]));
    const layout=layoutGraph(result.nodes);
    for(const n of result.nodes){
      const group=layout.groups.find(g=>g.id===n.layer),p=layout.relative[n.id];
      assert.ok(p.x>=0&&p.y>=0&&p.x+NODE_WIDTH<=group.width&&p.y+NODE_HEIGHT<=group.height,n.id);
    }
  }
});
test('SVG and Mermaid export all visible nodes/relationships and escape source text',()=>{
  const layout=layoutGraph(snapshot.nodes);
  const diagram=diagramSVG(snapshot.nodes,snapshot.edges,layout.positions,layout.groups);
  assert.ok(diagram.width>0&&diagram.height>0);
  assert.doesNotMatch(diagram.svg,/NaN|undefined|foreignObject|<script/);
  assert.equal((diagram.svg.match(/marker-end=/g)||[]).length,snapshot.edges.length);
  const hostile={...snapshot.nodes[0],title:'<script>&"test"</script>'};
  const escaped=diagramSVG([hostile],[],{[hostile.id]:{x:0,y:0}},[]);
  assert.doesNotMatch(escaped.svg,/<script/);assert.match(escaped.svg,/&lt;script&gt;/);
  const mermaid=mermaidDiagram(snapshot.nodes,snapshot.edges);
  assert.match(mermaid,/flowchart LR/);assert.match(mermaid,/-\.->/);
});
