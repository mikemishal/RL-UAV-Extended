import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { SimplifiedApp } from './SimplifiedApp';
import './styles.css';

const root = document.getElementById('root');
if (!root) throw new Error('Application root is missing.');

// Simplified two-scenario UI is the default root page; the full historical interface remains available,
// unchanged, at /legacy for development/research use (brief section 24).
const isLegacy = window.location.pathname.replace(/\/+$/, '') === '/legacy';

createRoot(root).render(<StrictMode>{isLegacy ? <App /> : <SimplifiedApp />}</StrictMode>);
