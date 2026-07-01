import { NavLink } from 'react-router-dom';

import { ConnectionIndicator } from '@/components/common/ConnectionIndicator';
import { useAnomalyStream } from '@/hooks/useAnomalyStream';

const LINKS = [
  { to: '/', label: 'Monitor', code: 'LIVE', icon: '◉', end: true },
  { to: '/eventos', label: 'Eventos', code: 'EVTS', icon: '▤', end: false },
  { to: '/busqueda', label: 'Búsqueda', code: 'SRCH', icon: '⌕', end: false },
  { to: '/offline', label: 'Offline', code: 'ANLZ', icon: '∿', end: false },
];

export function NavBar() {
  const { status, reconnectNow } = useAnomalyStream();
  return (
    <nav
      aria-label="Navegación principal"
      className="flex shrink-0 flex-col gap-1 border-b border-line bg-surface-2 p-4 md:w-60 md:border-b-0 md:border-r"
    >
      <div className="mb-5 hidden items-center gap-2 px-1 md:flex">
        <span className="grid h-8 w-8 place-items-center rounded-md border border-primary/40 bg-primary/10 font-display text-primary shadow-glow-primary">
          ◈
        </span>
        <div className="leading-tight">
          <div className="font-display text-sm font-bold uppercase tracking-widest text-ink">
            DSP·AD
          </div>
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-dim">
            mission control
          </div>
        </div>
      </div>

      <div className="flex flex-row gap-1 md:flex-col">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-md px-3 py-2 font-mono text-sm transition-colors ${
                isActive
                  ? 'bg-primary/10 text-primary ring-1 ring-inset ring-primary/30'
                  : 'text-muted hover:bg-white/5 hover:text-ink'
              }`
            }
          >
            <span aria-hidden className="text-base leading-none">
              {l.icon}
            </span>
            <span className="hidden sm:inline">{l.label}</span>
            <span className="ml-auto hidden text-[0.6rem] tracking-widest text-dim md:inline">
              {l.code}
            </span>
          </NavLink>
        ))}
      </div>

      <div className="mt-auto hidden px-1 pt-5 md:block">
        <ConnectionIndicator status={status} onReconnect={reconnectNow} />
      </div>
    </nav>
  );
}
