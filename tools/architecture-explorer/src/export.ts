import { colorFor, NODE_HEIGHT, NODE_WIDTH } from './graph.ts';
import type { GroupLayout, Point } from './graph.ts';
import type { Connection, Module } from './types.ts';

export const escapeXML = (text: string) => text.replace(/[&<>"']/g,c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]!));
export function diagramSVG(nodes: Module[], edges: Connection[], positions: Record<string,Point>, groups: GroupLayout[]) {
  const width = Math.max(720,...groups.map(g=>g.x+g.width),...nodes.map(n=>positions[n.id].x+NODE_WIDTH))+80;
  const height = Math.max(350,...groups.map(g=>g.y+g.height),...nodes.map(n=>positions[n.id].y+NODE_HEIGHT))+120;
  const groupMarkup = groups.map(g=>`<rect x="${g.x+40}" y="${g.y+70}" width="${g.width}" height="${g.height}" rx="16" fill="#191c23" stroke="${colorFor(g.id)}" stroke-opacity=".3"/><text x="${g.x+64}" y="${g.y+108}" fill="${colorFor(g.id)}" font-size="18" font-weight="600">${escapeXML(g.id.toUpperCase())}</text>`).join('');
  const edgeMarkup = edges.map(e=>{
    const a=positions[e.source],b=positions[e.target];
    const x1=a.x+NODE_WIDTH+40,y1=a.y+NODE_HEIGHT/2+70,x2=b.x+40,y2=b.y+NODE_HEIGHT/2+70;
    const bend=Math.max(50,Math.abs(x2-x1)/2);
    return `<path d="M${x1},${y1} C${x1+bend},${y1} ${x2-bend},${y2} ${x2},${y2}" fill="none" stroke="${e.conditional?'#d9a2f0':e.kind==='foreign-key'?'#65d7b0':'#8d96a8'}" stroke-width="1.5" ${e.conditional?'stroke-dasharray="7 5"':''} marker-end="url(#arrow)"><title>${escapeXML(e.label)}</title></path>`;
  }).join('');
  const nodeMarkup = nodes.map(n=>{
    const p=positions[n.id],x=p.x+40,y=p.y+70;
    const lines=n.title.length>29 ? [n.title.slice(0,n.title.lastIndexOf(' ',29)),n.title.slice(n.title.lastIndexOf(' ',29)+1)] : [n.title];
    return `<g><title>${escapeXML(n.description)}</title><rect x="${x}" y="${y}" width="${NODE_WIDTH}" height="${NODE_HEIGHT}" rx="12" fill="#252932" stroke="${colorFor(n.layer)}"/><rect x="${x+14}" y="${y+18}" width="5" height="26" rx="2" fill="${colorFor(n.layer)}"/>${lines.map((s,i)=>`<text x="${x+30}" y="${y+31+i*20}" fill="#eff1f7" font-size="14" font-weight="600">${escapeXML(s)}</text>`).join('')}<text x="${x+18}" y="${y+85}" fill="#b6bdcb" font-size="11">${escapeXML(n.technologies[0])} · ${n.files.length} files · ${escapeXML(n.status)}</text></g>`;
  }).join('');
  return {width,height,svg:`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#99a1b3"/></marker></defs><rect width="100%" height="100%" fill="#111318"/><g font-family="Arial, sans-serif"><text x="40" y="35" fill="#ffffff" font-size="22" font-weight="700">Filmory / Architecture Explorer</text><text x="40" y="55" fill="#adb4c5" font-size="12">Static source snapshot · ${nodes.length} nodes / ${edges.length} relationships · dashed = conditional · green arrows = declared FK</text>${groupMarkup}${edgeMarkup}${nodeMarkup}</g></svg>`};
}
export function mermaidDiagram(nodes: Module[], edges: Connection[]) {
  const clean = (s: string) => s.replace(/["\[\]<>\r\n]/g,' ').replace(/&/g,'and');
  return ['flowchart LR',...nodes.map(n=>`  ${n.id.replace(/-/g,'_')}["${clean(n.title)}"]`),
    ...edges.map(e=>`  ${e.source.replace(/-/g,'_')} ${e.conditional?'-.->':'-->'}|"${clean(e.label)}"| ${e.target.replace(/-/g,'_')}`)].join('\n');
}
export function download(content: BlobPart, filename: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content],{type:mime}));
  const link=document.createElement('a'); link.href=url; link.download=filename; link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export async function downloadPNG(diagram: ReturnType<typeof diagramSVG>) {
  const url=URL.createObjectURL(new Blob([diagram.svg],{type:'image/svg+xml'}));
  try {
    const image=new Image(); image.src=url;
    await new Promise<void>((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error('Could not render the exported SVG.'));});
    const scale=Math.min(2,4096/Math.max(diagram.width,diagram.height));
    const canvas=document.createElement('canvas');canvas.width=Math.ceil(diagram.width*scale);canvas.height=Math.ceil(diagram.height*scale);
    const ctx=canvas.getContext('2d'); if (!ctx) throw new Error('Canvas is unavailable. Use SVG export.');
    ctx.drawImage(image,0,0,canvas.width,canvas.height);
    const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('PNG encoding failed.')),'image/png'));
    download(blob,'filmory-architecture.png','image/png');
  } finally { URL.revokeObjectURL(url); }
}
