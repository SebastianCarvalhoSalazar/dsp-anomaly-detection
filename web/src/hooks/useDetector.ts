import { useMutation } from '@tanstack/react-query';

import { resetDetector } from '@/api/endpoints';

/** Pide al pipeline reiniciar su detector (vuelve a la fase de warmup). */
export function useResetDetector() {
  return useMutation({ mutationFn: () => resetDetector() });
}
