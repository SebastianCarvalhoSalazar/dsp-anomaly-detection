import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getFusionConfig, resetDetector, setFusionConfig } from '@/api/endpoints';
import { queryKeys } from '@/api/queryKeys';
import type { FusionConfig } from '@/api/types';

export function useFusionConfig() {
  return useQuery({
    queryKey: queryKeys.fusionConfig(),
    queryFn: getFusionConfig,
  });
}

export function useSetFusionConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cfg: FusionConfig) => setFusionConfig(cfg),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.fusionConfig() }),
  });
}

export function useResetDetector() {
  return useMutation({ mutationFn: () => resetDetector() });
}
