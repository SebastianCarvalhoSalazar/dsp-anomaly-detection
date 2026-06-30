import { API_BASE_URL } from './client';

export const audioUrl = (id: number): string => `${API_BASE_URL}/events/${id}/audio`;

export const frameUrl = (id: number, annotated = false): string =>
  `${API_BASE_URL}/events/${id}/frame${annotated ? '/annotated' : ''}`;
