import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  deleteAllEvents,
  deleteEvent,
  getEvent,
  listEvents,
  type ListEventsParams,
} from '@/api/endpoints';
import { queryKeys } from '@/api/queryKeys';

export function useEvents(params: ListEventsParams) {
  return useQuery({
    queryKey: queryKeys.events(params),
    queryFn: () => listEvents(params),
  });
}

export function useEvent(id: number, enabled = true) {
  return useQuery({
    queryKey: queryKeys.event(id),
    queryFn: () => getEvent(id),
    enabled,
  });
}

export function useDeleteEvent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteEvent(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['events'] }),
  });
}

export function useDeleteAllEvents() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => deleteAllEvents(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['events'] }),
  });
}
