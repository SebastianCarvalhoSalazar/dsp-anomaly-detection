import { useQuery } from '@tanstack/react-query';

import { getOfflineAnalysis } from '@/api/endpoints';
import { queryKeys } from '@/api/queryKeys';

export function useOfflineAnalysis(id: number | null, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.offlineAnalysis(id ?? -1),
    queryFn: () => getOfflineAnalysis(id as number),
    enabled: enabled && id !== null,
  });
}
