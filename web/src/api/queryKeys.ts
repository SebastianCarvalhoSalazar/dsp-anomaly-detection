import type { ListEventsParams } from './endpoints';

export const queryKeys = {
  events: (params: ListEventsParams) => ['events', params] as const,
  event: (id: number) => ['event', id] as const,
  offlineAnalysis: (id: number) => ['offlineAnalysis', id] as const,
  similarByEvent: (id: number, k: number) => ['similarByEvent', id, k] as const,
  fusionConfig: () => ['fusionConfig'] as const,
};
