// In development (npm run dev), requests go to the local FastAPI server.
// In production (built & served by FastAPI from the same origin), the
// API lives at the same host, so we use a relative path (empty string).
export const API_BASE_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''
