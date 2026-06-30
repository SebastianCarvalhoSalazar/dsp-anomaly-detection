export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
).replace(/\/$/, '');

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: BodyInit | null;
  /** JSON serializable; mutually exclusive with `body`. */
  json?: unknown;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { json, headers, ...rest } = opts;
  const init: RequestInit = { ...rest };
  const finalHeaders = new Headers(headers);
  finalHeaders.set('Accept', 'application/json');

  if (json !== undefined) {
    finalHeaders.set('Content-Type', 'application/json');
    init.body = JSON.stringify(json);
  }
  init.headers = finalHeaders;

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (err) {
    throw new ApiError(0, `No se pudo conectar con la API: ${(err as Error).message}`);
  }

  if (!res.ok) {
    const body = await parseBody(res);
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    if (res.status === 413) {
      throw new ApiError(413, 'El archivo supera el límite de 10 MB.', body);
    }
    throw new ApiError(res.status, detail || `Error HTTP ${res.status}`, body);
  }

  if (res.status === 204) return undefined as T;
  return (await parseBody(res)) as T;
}
