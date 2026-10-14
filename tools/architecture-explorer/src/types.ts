export type Layer = 'frontend' | 'backend' | 'database' | 'external' | 'offline';
export interface Evidence { path: string; line: number; anchor: string; excerpt: string }
export interface Route { route: string; file: string; line: number }
export interface Endpoint extends Route { method: string; handler: string; auth: string }
export interface Table { name: string; className: string; file: string; line: number; columns: {name: string; line: number}[]; foreignKeys: {column: string; target: string; line: number}[] }
export interface Module {
  id: string; title: string; layer: Layer; description: string; technologies: string[];
  files: string[]; evidence: Evidence[]; note: string; status: string;
  routes: Route[]; endpoints: Endpoint[]; tables: Table[];
}
export interface Connection { id: string; source: string; target: string; label: string; kind: string; conditional: boolean; note: string; evidence: Evidence[] }
export interface Walkthrough { id: string; title: string; summary: string; steps: string[][] }
export interface SourceFile { path: string; lines: number; sha256: string; imports: {module: string; line: number}[]; symbols: {name: string; line: number}[] }
export interface Snapshot {
  schemaVersion: number; root: string; repository: string; scope: string;
  nodes: Module[]; edges: Connection[]; walkthroughs: Walkthrough[];
  files: SourceFile[]; routes: Route[]; endpoints: Endpoint[]; tables: Table[];
  limitations: string[]; dependencies: {frontend: {dependencies: Record<string,string>; devDependencies: Record<string,string>}; backend: string[]};
  activeModelConfig: Record<string, unknown>;
}
