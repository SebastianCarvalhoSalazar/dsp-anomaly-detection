import { useQuery } from '@tanstack/react-query';

import { getSpectrogram } from '@/api/endpoints';
import { queryKeys } from '@/api/queryKeys';

/** Espectrograma mel de un evento; solo se pide cuando `enabled` es true. */
export function useSpectrogram(id: number, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.spectrogram(id),
    queryFn: () => getSpectrogram(id),
    enabled,
    staleTime: Infinity, // el audio de un evento no cambia
  });
}
