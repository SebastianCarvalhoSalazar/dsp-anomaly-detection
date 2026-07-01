import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { FusionConfig, FusionStrategy } from '@/api/types';

interface FusionDraftState extends FusionConfig {
  setStrategy: (s: FusionStrategy) => void;
  setAudioWeight: (w: number) => void;
  setGates: (g: boolean) => void;
  hydrate: (cfg: FusionConfig) => void;
}

/**
 * Borrador local de la configuración de fusión. Persiste entre navegaciones de
 * ruta (en memoria) y entre recargas (localStorage), resolviendo el bug B4 sin
 * los workarounds de estado de Streamlit.
 */
export const useFusionDraft = create<FusionDraftState>()(
  persist(
    (set) => ({
      strategy: 'weighted',
      audio_weight: 0.5,
      gates: false,
      setStrategy: (strategy) => set({ strategy }),
      setAudioWeight: (audio_weight) => set({ audio_weight }),
      setGates: (gates) => set({ gates }),
      hydrate: (cfg) => set({ ...cfg }),
    }),
    { name: 'fusion-draft' },
  ),
);
