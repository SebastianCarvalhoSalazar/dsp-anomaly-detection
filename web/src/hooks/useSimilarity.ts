import { useMutation, useQuery } from '@tanstack/react-query';

import { searchSimilarByEvent, searchSimilarUpload } from '@/api/endpoints';
import { queryKeys } from '@/api/queryKeys';
import type { Modality } from '@/api/types';

export function useSimilarByEvent(id: number | null, k: number, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.similarByEvent(id ?? -1, k),
    queryFn: () => searchSimilarByEvent(id as number, k),
    enabled: enabled && id !== null,
  });
}

export function useSimilarUpload() {
  return useMutation({
    mutationFn: (vars: { file: File; modality: Modality; k: number }) =>
      searchSimilarUpload(vars.file, vars.modality, vars.k),
  });
}
