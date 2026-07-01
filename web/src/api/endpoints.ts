import { request } from './client';
import type {
  AnomalyScoreMessage,
  EventResponse,
  FusionConfig,
  Modality,
  OfflineAnalysisResponse,
  SimilarEventResponse,
} from './types';

export interface ListEventsParams {
  limit?: number;
  offset?: number;
  min_score?: number;
}

function qs(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : '';
}

export const listEvents = (params: ListEventsParams = {}): Promise<EventResponse[]> =>
  request(`/events/${qs({ ...params })}`);

export const getEvent = (id: number): Promise<EventResponse> => request(`/events/${id}`);

export const deleteEvent = (id: number): Promise<void> =>
  request(`/events/${id}`, { method: 'DELETE' });

export const deleteAllEvents = (): Promise<void> => request('/events/', { method: 'DELETE' });

export const getOfflineAnalysis = (id: number): Promise<OfflineAnalysisResponse> =>
  request(`/events/${id}/offline_analysis`);

export const searchSimilarByEvent = (id: number, k = 5): Promise<SimilarEventResponse[]> =>
  request(`/search/similar/by-event/${id}${qs({ k })}`);

export const searchSimilarUpload = (
  file: File,
  modality: Modality = 'audio',
  k = 5,
): Promise<SimilarEventResponse[]> => {
  const form = new FormData();
  form.append('file', file);
  // No fijar Content-Type: el navegador añade el boundary del multipart.
  return request(`/search/similar${qs({ modality, k })}`, { method: 'POST', body: form });
};

export const getFusionConfig = (): Promise<FusionConfig> => request('/internal/fusion-config');

export const setFusionConfig = (cfg: FusionConfig): Promise<unknown> =>
  request('/internal/fusion-config', { method: 'POST', json: cfg });

export const resetDetector = (): Promise<unknown> =>
  request('/internal/reset-detector', { method: 'POST' });

export type { AnomalyScoreMessage };
