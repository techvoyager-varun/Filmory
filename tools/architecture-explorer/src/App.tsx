import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ReactFlow, Background, Controls, MiniMap, Handle, Position, MarkerType, useNodesState, useReactFlow } from '@xyflow/react';
import type { Node, NodeProps, Edge } from '@xyflow/react';
import { ArrowDownToLine, ArrowLeft, ArrowRight, BookOpen, Boxes, Check, ChevronRight, CircleHelp, Copy, Database, FileCode2, Filter, Focus, GitBranch, Globe2, Layers3, LayoutGrid, Monitor, Play, Search, Server, ShieldCheck, Workflow, X } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import raw from './architecture.json';
import type { Connection, Evidence, Layer, Module, Snapshot, SourceFile } from './types';
import { absoluteFile, colorFor, filterGraph, layers, layoutGraph, NODE_HEIGHT, NODE_WIDTH } from './graph';
import { diagramSVG, download, downloadPNG, mermaidDiagram } from './export';

const snapshot = raw as Snapshot;
const icons: Record<Layer, LucideIcon> = {frontend:Monitor,backend:Server,database:Database,external:Globe2,offline:Boxes};
type FlowModule = Node<{module:Module; muted:boolean; guided:boolean}, 'module'>;
type FlowGroup = Node<{title:string; subtitle:string; color:string}, 'section'>;
type FlowNode = FlowModule | FlowGroup;
function ModuleNode({data,selected}: NodeProps<FlowModule>) {
  const {module:n}=data, Icon=icons[n.layer];
  return <div className={`module-node ${selected?'is-selected':''} ${data.muted?'is-muted':''} ${data.guided?'is-guided':''}`} style={{'--accent':colorFor(n.layer)} as React.CSSProperties}>
    <Handle type="target" position={Position.Left} /><div className="module-top"><span className="module-icon"><Icon size={21}/></span><span className="module-title">{n.title}</span><span className={`status-dot ${n.status}`} title={n.status}/></div>
    <div className="module-tech">{n.technologies[0]}</div><div className="module-footer"><FileCode2 size={12}/>{n.files.length} files<span>{n.endpoints.length?`${n.endpoints.length} endpoints`:n.tables.length?`${n.tables.length} tables`:n.status === 'source'?'source verified':n.status}</span></div>
    <Handle type="source" position={Position.Right}/>
  </div>;
}
function SectionNode({data}:NodeProps<FlowGroup>) {return <div className="section-node" style={{'--accent':data.color} as React.CSSProperties}><div className="section-label"><span/>{data.title}<small>{data.subtitle}</small></div></div>;}
const nodeTypes={module:ModuleNode,section:SectionNode};
const allLayers=layers.map(l=>l.id);

export function App() {
  const [tab,setTab]=useState('graph');
  const [query,setQuery]=useState('');
  const [enabled,setEnabled]=useState<Layer[]>(allLayers);
  const [status,setStatus]=useState('all');
  const [relation,setRelation]=useState('runtime');
  const [view,setView]=useState('overview');
  const [selected,setSelected]=useState<string|null>(null);
  const [selectedEdge,setSelectedEdge]=useState<string|null>(null);
  const [selectedFile,setSelectedFile]=useState<string|null>(null);
  const [walkId,setWalkId]=useState('');
  const [step,setStep]=useState(0);
  const [tourChoice,setTourChoice]=useState('recommendations');
  const [notice,setNotice]=useState('');
  const [exporting,setExporting]=useState(false);
  const [exportOpen,setExportOpen]=useState(false);
  const [filesPage,setFilesPage]=useState(0);
  const noticeTimer=useRef<ReturnType<typeof setTimeout>|null>(null);
  const {fitView,getNodes,toObject}=useReactFlow<FlowNode>();
  const focus=snapshot.walkthroughs.find(w=>w.id===view);
  const focusedIds=useMemo(()=>focus ? [...new Set(focus.steps.map(s=>s[0]).concat(view==='assistant'?['http','app','gemini','catalog-db','ranking']:view==='auth'?['app']:view==='activity'?['http','app','identity-db']:view==='training'?['recommender']:['app','bundle','identity-db','activity-db']))] : [],[focus,view]);
  const visible=useMemo(()=>filterGraph(snapshot.nodes,snapshot.edges,enabled,query,status,relation,focusedIds),[enabled,query,status,relation,focusedIds]);
  const layout=useMemo(()=>layoutGraph(visible.nodes),[visible.nodes]);
  const walk=snapshot.walkthroughs.find(w=>w.id===walkId);
  const activeStepId=walk?.steps[step]?.[0];
  const [nodes,setNodes,onNodesChange]=useNodesState<FlowNode>([]);
  const edges:Edge[]=useMemo(()=>visible.edges.map(e=>({id:e.id,source:e.source,target:e.target,type:'smoothstep',
    animated:Boolean(walk && e.source===activeStepId),selected:e.id===selectedEdge,
    markerEnd:{type:MarkerType.ArrowClosed,color:e.conditional?'#c896d9':e.kind==='foreign-key'?'#65d7b0':'#838d9f'},
    style:{stroke:e.id===selectedEdge?'#ff927a':e.conditional?'#bc8fce':e.kind==='foreign-key'?'#65d7b0':'#7a8496',strokeWidth:e.id===selectedEdge?3:1.5,strokeDasharray:e.conditional?'6 5':undefined,opacity:selected && e.source!==selected && e.target!==selected ? .3:.85},
    label:e.id===selectedEdge?e.label:undefined,labelStyle:{fill:'#f2f3f7',fontSize:11},labelBgStyle:{fill:'#252932'},labelBgPadding:[8,5],
  })),[visible.edges,walk,activeStepId,selectedEdge,selected]);
  const notify=useCallback((message:string)=>{setNotice(message);if(noticeTimer.current)clearTimeout(noticeTimer.current);noticeTimer.current=setTimeout(()=>setNotice(''),4500);},[]);
  useEffect(()=>()=>{if(noticeTimer.current)clearTimeout(noticeTimer.current);},[]);
  const closeInspector=()=>{setSelected(null);setSelectedEdge(null);setSelectedFile(null);};
  const pickNode=useCallback((id:string, zoom=true, expand=true)=>{if(expand && !getNodes().some(n=>n.id===id)){setView('overview');setQuery('');setEnabled(allLayers);setStatus('all');setWalkId('');}setSelected(id);setSelectedEdge(null);setSelectedFile(null);setTab('graph');if(zoom)window.setTimeout(()=>void fitView({nodes:[{id}],padding:.8,maxZoom:.95,duration:450}),250);},[fitView,getNodes]);
  useEffect(()=>{
    const next:FlowNode[]=[...layout.groups.map(g=>({id:'group-'+g.id,type:'section' as const,position:{x:g.x,y:g.y},data:{title:layers.find(l=>l.id===g.id)!.title,subtitle:layers.find(l=>l.id===g.id)!.subtitle,color:colorFor(g.id)},style:{width:g.width,height:g.height},draggable:false,selectable:false,connectable:false})),
      ...visible.nodes.map(n=>({id:n.id,type:'module' as const,parentId:'group-'+n.layer,extent:'parent' as const,position:layout.relative[n.id],width:NODE_WIDTH,height:NODE_HEIGHT,data:{module:n,muted:false,guided:false}}))];
    setNodes(next);
    const timeout=window.setTimeout(()=>void fitView({padding:.16,duration:350,minZoom:.15,maxZoom:.9}),100);
    return ()=>window.clearTimeout(timeout);
  },[visible.nodes,layout,setNodes,fitView]);
  useEffect(()=>{
    setNodes(previous=>previous.map(n=>n.type==='module'?{...n,selected:n.id===selected,data:{...n.data,muted:Boolean(selected && n.id!==selected && !visible.edges.some(e=>(e.source===selected && e.target===n.id)||(e.target===selected && e.source===n.id))),guided:n.id===activeStepId}}:n));
  },[selected,activeStepId,visible.edges,setNodes]);
  useEffect(()=>{
    if(activeStepId){pickNode(activeStepId,false,false);const timeout=window.setTimeout(()=>void fitView({nodes:[{id:activeStepId}],padding:.65,maxZoom:.9,duration:450}),220);return ()=>window.clearTimeout(timeout);}
  },[activeStepId,pickNode,fitView]);
  useEffect(()=>{setFilesPage(0);},[query]);
  useEffect(()=>{
    if(selected && !visible.nodes.some(n=>n.id===selected))setSelected(null);
    if(selectedEdge && !visible.edges.some(e=>e.id===selectedEdge))setSelectedEdge(null);
  },[visible,selected,selectedEdge]);
  const reset=()=>{setEnabled(allLayers);setQuery('');setStatus('all');setRelation('runtime');setView('overview');setWalkId('');closeInspector();};
  const changeView=(id:string)=>{setView(id);setWalkId('');setQuery('');setEnabled(allLayers);setStatus('all');closeInspector();};
  const startWalk=()=>{setWalkId(tourChoice);setStep(0);setView(tourChoice);setQuery('');setEnabled(allLayers);setStatus('all');setRelation('runtime');setTab('graph');};
  const module=snapshot.nodes.find(n=>n.id===selected);
  const connection=snapshot.edges.find(e=>e.id===selectedEdge);
  const file=snapshot.files.find(f=>f.path===selectedFile);
  const copy=async(text:string)=>{try{await navigator.clipboard.writeText(text);notify('Copied to clipboard.');}catch{notify('Clipboard unavailable. Select and copy the displayed path.');}};
  async function exportDiagram(format:string) {
    if(!visible.nodes.length){notify('No visible nodes to export. Clear a filter first.');return;}
    setExporting(true);setExportOpen(false);
    try {
      const actualNodes=getNodes();const positions={...layout.positions};
      visible.nodes.forEach(n=>{const actual=actualNodes.find(a=>a.id===n.id),parent=actualNodes.find(a=>a.id===actual?.parentId);if(actual)positions[n.id]={x:actual.position.x+(parent?.position.x??0),y:actual.position.y+(parent?.position.y??0)};});
      const diagram=diagramSVG(visible.nodes,visible.edges,positions,layout.groups);
      if(format==='svg')download(diagram.svg,'filmory-architecture.svg','image/svg+xml');
      if(format==='png')await downloadPNG(diagram);
      if(format==='mermaid')download(mermaidDiagram(visible.nodes,visible.edges),'filmory-architecture.mmd','text/plain');
      if(format==='json')download(JSON.stringify({snapshot,currentView:{view,query,enabled,status,relation,visibleNodeIds:visible.nodes.map(n=>n.id),visibleEdgeIds:visible.edges.map(e=>e.id)},reactFlow:toObject()},null,2),'filmory-architecture.json','application/json');
      notify(`Exported ${format.toUpperCase()}${format==='json'?' with complete source snapshot':' of the visible diagram'}.`);
    }catch(error){notify(error instanceof Error?error.message:'Export failed.');}finally{setExporting(false);}
  }
  const inventory=snapshot.files.filter(f=>[f.path,...f.imports.map(i=>i.module),...f.symbols.map(s=>s.name)].join(' ').toLowerCase().includes(query.trim().toLowerCase()));
  const inspector=Boolean(module || connection || file);
  return <div className="app">
    <header className="topbar"><div className="brand-mark"><Workflow size={24}/></div><div className="brand"><strong>filmory<span> / </span>architecture</strong><small>Explore the code behind the experience</small></div><span className="source-badge"><span/>Source snapshot</span><div className="top-actions"><button className="subtle" onClick={()=>{setTab('evidence');closeInspector();}}><ShieldCheck size={15}/>Evidence & scope</button><div className="export-menu"><button className="accent" onClick={()=>setExportOpen(!exportOpen)} disabled={exporting}><ArrowDownToLine size={15}/>{exporting?'Exportingâ€¦':'Export diagram'}</button>{exportOpen && <div className="dropdown">{['svg','png','mermaid','json'].map(f=><button key={f} onClick={()=>void exportDiagram(f)}>{f.toUpperCase()}<small>{f==='json'?'Full snapshot + layout':f==='mermaid'?'Editable diagram source':'Visible nodes & connections'}</small></button>)}<button onClick={()=>setExportOpen(false)}>Close</button></div>}</div></div></header>
    <div className="workspace">
      <aside className="sidebar"><div className="sidebar-heading"><Layers3 size={16}/><strong>WORKSPACE</strong></div><nav aria-label="Explorer sections"><button className={tab==='graph'?'active':''} onClick={()=>setTab('graph')}><GitBranch size={16}/>Architecture map<span>{snapshot.nodes.length}</span></button><button className={tab==='files'?'active':''} onClick={()=>{setTab('files');closeInspector();}}><FileCode2 size={16}/>Source inventory<span>{snapshot.files.length}</span></button></nav>
        <label className="search-field"><Search size={16}/><input aria-label="Search architecture" value={query} placeholder={tab==='files'?'Find a path or symbolâ€¦':'Find node, path or endpointâ€¦'} onChange={e=>{setQuery(e.target.value);if(tab==='graph')setView('overview');if(walk)setWalkId('');}}/>{query&&<button aria-label="Clear search" onClick={()=>setQuery('')}><X size={14}/></button>}</label>
        {tab==='graph' && <><div className="eyebrow"><Filter size={13}/>Visible layers</div><div className="layer-filters">{layers.map(l=><label key={l.id}><input type="checkbox" checked={enabled.includes(l.id)} onChange={()=>{setWalkId('');setEnabled(enabled.includes(l.id)?enabled.filter(x=>x!==l.id):[...enabled,l.id]);}}/><span style={{background:l.color}}/>{l.title}<small>{snapshot.nodes.filter(n=>n.layer===l.id).length}</small></label>)}</div><label className="select-label">Module status<select aria-label="Module status" value={status} onChange={e=>{setWalkId('');setStatus(e.target.value);}}><option value="all">All statuses</option><option value="active">Exclude unused modules</option><option value="conditional">Conditional integrations</option><option value="inactive">Unused / schema only</option><option value="artifact">Artifact metadata</option></select></label><label className="select-label">Relationships<select aria-label="Relationships" value={relation} onChange={e=>setRelation(e.target.value)}><option value="runtime">Calls, composition & artifacts</option><option value="all">All, including declared FKs</option><option value="foreign-key">Declared foreign keys only</option><option value="conditional">Conditional calls only</option></select></label><div className="eyebrow">Visible modules <span>{visible.nodes.length}</span></div><div className="node-library">{visible.nodes.map(n=>{const Icon=icons[n.layer];return <button key={n.id} className={selected===n.id?'chosen':''} onClick={()=>pickNode(n.id)}><Icon size={14} color={colorFor(n.layer)}/><span>{n.title}</span><ChevronRight size={12}/></button>;})}{!visible.nodes.length&&<p className="empty-text">No matches. Clear search or reset filters.</p>}</div></>}
        {tab!=='graph'&&<div className="sidebar-note"><BookOpen size={22}/><p>Read the actual paths, symbols, route declarations and dependency versions.</p><small>The inventory is generated locally. Secrets, raw data and dependency internals are excluded.</small></div>}
        <div className="sidebar-bottom"><span className="local-dot"/>Independent local viewer<button onClick={reset}>Reset map</button></div>
      </aside>
      <main className="main-panel">
        {tab==='graph' ? <><div className="canvas-toolbar"><div><span className="eyebrow">ARCHITECTURE MAP</span><h1>{focus?focus.title:'The system, connected.'}</h1></div><label className="view-select"><LayoutGrid size={15}/><select aria-label="Graph view" value={view} onChange={e=>changeView(e.target.value)}><option value="overview">Complete architecture</option>{snapshot.walkthroughs.map(w=><option key={w.id} value={w.id}>{w.title}</option>)}</select></label><button className="subtle" onClick={()=>void fitView({padding:.16,duration:450,maxZoom:.9})}><Focus size={15}/>Fit view</button></div>
          <div className="canvas"><ReactFlow<FlowNode> nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onNodeClick={(_,n)=>{if(n.type==='module')pickNode(n.id,false);}} onEdgeClick={(_,e)=>{setSelectedEdge(e.id);setSelected(null);setSelectedFile(null);}} onPaneClick={closeInspector} nodesConnectable={false} edgesReconnectable={false} deleteKeyCode={null} minZoom={.12} maxZoom={1.8} fitView colorMode="dark"><Background color="#3a3e4b" gap={22} size={1}/><Controls showInteractive={false}/><MiniMap nodeColor={n=>n.type==='module'?colorFor((n.data.module as Module).layer):'#222630'} maskColor="rgba(10,12,16,.65)" pannable zoomable/></ReactFlow>
            <div className="canvas-caption"><span>{visible.nodes.length} modules</span><span>{visible.edges.length} relationships</span><span>Click a node or arrow to inspect</span></div><div className="legend"><span><i/>Source relationship</span><span><i className="dashed"/>Conditional branch</span>{relation==='all'||relation==='foreign-key'?<span><i className="fk"/>Declared foreign key</span>:null}</div>
            {!visible.nodes.length&&<div className="empty-canvas"><Search size={36}/><h2>No nodes match these filters</h2><button onClick={reset}>Reset map</button></div>}
          </div>
          <div className="walkthrough-bar">{walk ? <><div className="walk-step"><span className="walk-icon"><Play size={18}/></span><div><small>GUIDED WALKTHROUGH Â· {step+1} OF {walk.steps.length}</small><strong>{snapshot.nodes.find(n=>n.id===activeStepId)?.title}</strong><p>{walk.steps[step][1]}</p></div></div><div className="walk-controls"><button className="subtle" disabled={step===0} onClick={()=>setStep(step-1)} aria-label="Previous step"><ArrowLeft size={17}/></button><button className="accent" onClick={()=>step+1===walk.steps.length?setWalkId(''):setStep(step+1)}>{step+1===walk.steps.length?'Finish':'Next'}<ArrowRight size={15}/></button><button className="icon-button" onClick={()=>setWalkId('')} aria-label="Exit walkthrough"><X size={16}/></button></div></> : <><div className="walk-intro"><span className="walk-icon"><Workflow size={21}/></span><div><strong>Follow a request through Filmory</strong><small>Step through real calls, branches and stored data.</small></div></div><select aria-label="Choose walkthrough" value={tourChoice} onChange={e=>setTourChoice(e.target.value)}>{snapshot.walkthroughs.map(w=><option key={w.id} value={w.id}>{w.title}</option>)}</select><button className="accent" onClick={startWalk}><Play size={14}/>Start walkthrough</button></>}</div>
        </> : tab==='files' ? <section className="inventory-page"><div className="page-heading"><span className="eyebrow">REPOSITORY INVENTORY</span><h1>Every scanned source file.</h1><p>{inventory.length} matching files Â· imports and symbols are static declarations, not execution traces.</p></div><div className="file-table"><div className="file-table-head"><span>Path</span><span>Lines</span><span>Imports</span></div>{inventory.slice(filesPage*60,(filesPage+1)*60).map(f=><button key={f.path} onClick={()=>{setSelectedFile(f.path);setSelected(null);setSelectedEdge(null);}}><span><FileCode2 size={15}/>{f.path}</span><small>{f.lines}</small><small>{f.imports.length}</small></button>)}</div>{!inventory.length&&<p>No files match this search.</p>}<div className="pagination"><button disabled={filesPage===0} onClick={()=>setFilesPage(filesPage-1)}>Previous</button><span>Page {filesPage+1} / {Math.max(1,Math.ceil(inventory.length/60))}</span><button disabled={(filesPage+1)*60>=inventory.length} onClick={()=>setFilesPage(filesPage+1)}>Next</button></div></section> : <EvidencePage/>}
      </main>
      {inspector&&<aside className="inspector" aria-label="Selected item details"><div className="inspector-top"><span className="eyebrow">{file?'SOURCE FILE':connection?'RELATIONSHIP':'MODULE DETAILS'}</span><button className="icon-button" onClick={closeInspector} aria-label="Close details"><X size={18}/></button></div>{module?<ModuleDetails module={module} pick={pickNode} copy={copy}/>:connection?<ConnectionDetails connection={connection} pick={pickNode} copy={copy}/>:file?<FileDetails file={file} copy={copy}/>:null}</aside>}
    </div><div role="status" aria-live="polite" className={`toast ${notice?'show':''}`}><Check size={16}/>{notice}</div>
  </div>;
}

function EvidenceList({items,copy}:{items:Evidence[];copy:(text:string)=>Promise<void>}) {return <div className="evidence-list">{items.map((e,i)=><div key={i} className="evidence-card"><div><FileCode2 size={13}/><strong>{e.path}:{e.line}</strong><button aria-label={`Copy ${e.path}`} onClick={()=>void copy(absoluteFile(snapshot.root,e.path)+':'+e.line)}><Copy size={13}/></button></div><code>{e.excerpt}</code></div>)}</div>;}
function ModuleDetails({module:n,pick,copy}:{module:Module;pick:(id:string,zoom?:boolean)=>void;copy:(text:string)=>Promise<void>}) {
  const Icon=icons[n.layer];const links=snapshot.edges.filter(e=>e.source===n.id||e.target===n.id);
  return <><div className="detail-title"><span className="large-icon" style={{color:colorFor(n.layer)}}><Icon size={26}/></span><div><small>{layers.find(l=>l.id===n.layer)?.title} / {n.status}</small><h2>{n.title}</h2></div></div><p className="detail-description">{n.description}</p><div className="tags">{n.technologies.map(t=><span key={t}>{t}</span>)}</div>{n.note&&<div className="note"><CircleHelp size={16}/><p>{n.note}</p></div>}<h3>Source evidence</h3><EvidenceList items={n.evidence} copy={copy}/><h3>Real file paths <span>{n.files.length}</span></h3><p className="micro-copy mono">Root: {snapshot.root}</p><div className="path-list">{n.files.map(p=><button key={p} title={absoluteFile(snapshot.root,p)} onClick={()=>void copy(absoluteFile(snapshot.root,p))}><FileCode2 size={13}/><code>{p}</code><Copy size={12}/></button>)}</div>
    {n.routes.length>0&&<><h3>Frontend routes</h3>{n.routes.map(r=><div className="endpoint" key={r.route}><span>PAGE</span><code>{r.route}</code></div>)}</>}
    {n.endpoints.length>0&&<><h3>API declarations</h3>{n.endpoints.map(e=><div className="endpoint" key={e.method+e.route}><span>{e.method}</span><code>{e.route}</code><small>{e.auth==='optional'?'optional bearer; handler may require identity':e.auth} Â· line {e.line}</small></div>)}</>}
    {n.tables.map(t=><details className="table-details" key={t.name}><summary><Database size={14}/>{t.name}<small>{t.columns.length} columns</small></summary><p className="mono">{t.columns.map(c=>c.name).join(', ')}</p>{t.foreignKeys.map(f=><p className="mono" key={f.column}>{f.column} â†’ {f.target}</p>)}<small>Declared in {t.file}:{t.line}</small></details>)}
    {(n.id==='bundle'||n.id==='loader')&&<><h3>Selected model configuration</h3><pre className="model-config">{JSON.stringify(snapshot.activeModelConfig,null,2)}</pre><p className="micro-copy">Metadata read from the active bundle. Checkpoint bytes are not loaded by this viewer.</p></>}
    <h3>Data flow & dependencies <span>{links.length}</span></h3><p className="micro-copy">All recorded connections, including ones hidden by your current filters.</p>{links.map(e=><div className="connection-card" key={e.id}><button onClick={()=>pick(e.source===n.id?e.target:e.source)}><span>{e.target===n.id?'IN':'OUT'}</span>{snapshot.nodes.find(x=>x.id===(e.source===n.id?e.target:e.source))?.title}<ArrowRight size={13}/></button><p>{e.label}</p><small>{e.kind} Â· {e.conditional?'conditional':'declared source relationship'}</small>{e.note&&<p className="micro-copy">{e.note}</p>}<EvidenceList items={e.evidence} copy={copy}/></div>)}{!links.length&&<p className="micro-copy">No runtime relationship asserted for this module.</p>}</>;
}
function ConnectionDetails({connection:e,pick,copy}:{connection:Connection;pick:(id:string,zoom?:boolean)=>void;copy:(text:string)=>Promise<void>}) {return <><div className="detail-title"><GitBranch size={25}/><h2>{e.label}</h2></div><p className="detail-description">{e.kind} Â· {e.conditional?'Conditional source branch':'Source relationship'}</p><div className="flow-pair">{[e.source,e.target].map((id,i)=><button key={id} onClick={()=>pick(id)}><small>{i===0?'FROM':'TO'}</small>{snapshot.nodes.find(n=>n.id===id)?.title}<ArrowRight size={16}/></button>)}</div>{e.note&&<div className="note"><p>{e.note}</p></div>}<h3>Exact evidence</h3><EvidenceList items={e.evidence} copy={copy}/><p className="micro-copy">This arrow proves the stated source relationship. External availability and runtime execution are not implied.</p></>;}
function FileDetails({file:f,copy}:{file:SourceFile;copy:(text:string)=>Promise<void>}) {return <><h2 className="file-heading">{f.path}</h2><button className="subtle" onClick={()=>void copy(absoluteFile(snapshot.root,f.path))}><Copy size={14}/>Copy absolute path</button><p className="detail-description">{f.lines} lines Â· {f.imports.length} import declarations Â· {f.symbols.length} Python symbols</p><h3>SHA-256 of normalized source</h3><code className="hash">{f.sha256}</code><h3>Declared imports</h3>{f.imports.map((i,index)=><div className="symbol" key={index}><code>{i.module || '(relative module)'}</code><small>line {i.line}</small></div>)}<h3>Python classes / functions</h3>{f.symbols.map((s,index)=><div className="symbol" key={index}><code>{s.name}</code><small>line {s.line}</small></div>)}{!f.symbols.length&&<p className="micro-copy">TypeScript symbols are not parsed by this scanner.</p>}</>;}
function EvidencePage() {return <section className="evidence-page"><span className="eyebrow">EVIDENCE & SCOPE</span><h1>Read the architecture with its evidence.</h1><p className="lead">This explorer is a static, source-backed map of Filmory. A connection describes what the code declares, not whether a provider, database or model succeeded in a live request.</p><div className="stat-grid">{[[snapshot.files.length,'source files'],[snapshot.routes.length,'frontend routes'],[snapshot.endpoints.length,'API endpoints'],[snapshot.tables.length,'database tables']].map(([count,title])=><div key={title}><strong>{count}</strong><span>{title}</span></div>)}</div><h2>What remains unverified</h2>{snapshot.limitations.map((s,i)=><div className="scope-row" key={s}><span>{String(i+1).padStart(2,'0')}</span><p>{s}</p></div>)}<h2>Active model configuration</h2><p>The local pointer selects <code>20261001_clean_extended</code>. This metadata is inspected without loading checkpoint bytes.</p><pre>{JSON.stringify(snapshot.activeModelConfig,null,2)}</pre><h2>Declared dependencies</h2><div className="dependency-columns"><div><h3>Frontend dependencies</h3>{Object.entries(snapshot.dependencies.frontend.dependencies).map(([name,version])=><p className="dependency" key={name}><code>{name}</code><span>{version}</span></p>)}<details><summary>Frontend build / development</summary>{Object.entries(snapshot.dependencies.frontend.devDependencies).map(([name,version])=><p className="dependency" key={name}><code>{name}</code><span>{version}</span></p>)}</details></div><div><h3>Backend requirements</h3>{snapshot.dependencies.backend.filter(s=>s.trim()&&!s.startsWith('#')).map(s=><p className="mono" key={s}>{s}</p>)}</div></div><h2>Refreshing this map</h2><p>From <code>tools/architecture-explorer</code>, run <code>npm run scan</code> then <code>npm run verify</code>. The scanner fails if a curated relationship loses its source anchor. Review relationship meaning after code changes; matching text alone cannot certify unchanged semantics.</p><p className="micro-copy">Separate package, separate lockfile, no Filmory backend/API access. Visual node positions are temporary; exports capture the current layout. Connection editing is disabled.</p></section>;}
