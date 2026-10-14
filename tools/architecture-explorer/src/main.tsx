import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { App } from './App';
import './style.css';

createRoot(document.getElementById('root')!).render(<StrictMode><ReactFlowProvider><App /></ReactFlowProvider></StrictMode>);
