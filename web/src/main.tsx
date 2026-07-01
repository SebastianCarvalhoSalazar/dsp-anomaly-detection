import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import App from './App';
import { AnomalyStreamProvider } from './hooks/useAnomalyStream';
import './styles/index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <AnomalyStreamProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AnomalyStreamProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
