import type { ReactNode } from 'react';

import { NavBar } from './NavBar';

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-primary focus:px-3 focus:py-1 focus:text-bg"
      >
        Saltar al contenido
      </a>
      <NavBar />
      <main id="main" className="mx-auto w-full max-w-screen-2xl flex-1 p-4 md:p-8">
        {children}
      </main>
    </div>
  );
}
