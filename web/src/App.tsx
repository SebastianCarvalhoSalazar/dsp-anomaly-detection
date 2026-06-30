import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';

import { LoadingSkeleton } from '@/components/common/States';
import { AppShell } from '@/layout/AppShell';

const LiveMonitor = lazy(() => import('@/pages/LiveMonitor'));
const EventFeed = lazy(() => import('@/pages/EventFeed'));
const SimilaritySearch = lazy(() => import('@/pages/SimilaritySearch'));
const OfflineAnalysis = lazy(() => import('@/pages/OfflineAnalysis'));

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<LoadingSkeleton rows={5} />}>
        <Routes>
          <Route path="/" element={<LiveMonitor />} />
          <Route path="/eventos" element={<EventFeed />} />
          <Route path="/busqueda" element={<SimilaritySearch />} />
          <Route path="/offline" element={<OfflineAnalysis />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
