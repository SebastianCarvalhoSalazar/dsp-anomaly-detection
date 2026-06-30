import { NavLink } from 'react-router-dom';

import { ConnectionIndicator } from '@/components/common/ConnectionIndicator';
import { useAnomalyStream } from '@/hooks/useAnomalyStream';

const LINKS = [
  { to: '/', label: 'Monitor en vivo', icon: '📡', end: true },
  { to: '/eventos', label: 'Eventos', icon: '🗂️', end: false },
  { to: '/busqueda', label: 'Búsqueda', icon: '🔍', end: false },
  { to: '/offline', label: 'Análisis offline', icon: '📊', end: false },
];

export function NavBar() {
  const { status, reconnectNow } = useAnomalyStream();
  return (
    <nav
      aria-label="Navegación principal"
      className="flex shrink-0 flex-col gap-1 bg-ink p-4 md:w-64"
    >
      <div className="mb-4 px-2 text-center">
        <div className="text-3xl">📡</div>
        <div className="mt-1 font-bold text-white">DSP Anomaly</div>
        <div className="text-[0.65rem] uppercase tracking-widest text-slate-500">
          Sistema de detección
        </div>
      </div>

      <div className="flex flex-row gap-1 md:flex-col">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive
                  ? 'bg-primary/30 text-white'
                  : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`
            }
          >
            <span aria-hidden>{l.icon}</span>
            <span className="hidden sm:inline">{l.label}</span>
          </NavLink>
        ))}
      </div>

      <div className="mt-auto hidden px-2 pt-4 md:block">
        <ConnectionIndicator status={status} onReconnect={reconnectNow} />
      </div>
    </nav>
  );
}
