# CONVENTIONS.md

Coding standards for VoiceBot Demo. Claude Code must follow these in every file it touches.

---

## Python / FastAPI (Backend)

### General
- Python 3.14+
- All route handlers must be `async def`
- Use `pydantic-settings` for all config — never `os.getenv()` directly in routes
- All external API calls go through `services/` — never call Retell or Sarvam directly from routes

### Imports
```python
# Standard lib first, then third-party, then local
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.retell_service import RetellService
from config import settings
```

### Error Handling
- Use `HTTPException` with proper status codes
- Log errors with Python's `logging` module, not `print()`
- Wrap external API calls in try/except — never let Sarvam/Retell errors crash the server

```python
# Good
try:
    result = await retell.create_call(...)
except RetellAPIError as e:
    logger.error(f"Retell call failed: {e}")
    raise HTTPException(status_code=502, detail="Failed to initiate call")
```

### Pydantic Models
- All request bodies = Pydantic BaseModel
- All response shapes = Pydantic BaseModel
- Use `model_config = ConfigDict(from_attributes=True)` for ORM models

### Database
- Use `aiosqlite` for async SQLite access
- Keep SQL in `db.py` — no raw SQL in routes or services
- Always use parameterized queries (`?` placeholders)

---

## TypeScript / React (Frontend)

### General
- React 18 + TypeScript strict mode
- Functional components only — no class components
- Tailwind CSS for all styling — no inline styles, no CSS files unless absolutely needed
- Use `axios` for all API calls via `src/api/client.ts`

### Component Structure
```tsx
// Named export, not default for components (except pages)
export function CallCard({ call }: { call: CallRecord }) {
  // hooks first
  const [loading, setLoading] = useState(false)
  
  // handlers next
  const handleClick = () => { ... }
  
  // render last
  return (
    <div className="...">
      ...
    </div>
  )
}
```

### API Client
```typescript
// Always use the central client — never raw fetch
import { api } from '@/api/client'

const result = await api.initiateCall({ phone, goal })
```

### Type Safety
- Define TypeScript interfaces for all API response shapes in `src/types/`
- No `any` types — use `unknown` + type guard if needed

### Naming
- Components: `PascalCase`
- Functions/variables: `camelCase`
- Files: `PascalCase.tsx` for components, `camelCase.ts` for utils/services

---

## Git / File Hygiene

- `.env` is never committed — only `.env.example`
- `requirements.txt` pinned versions only (use `pip freeze`)
- All secrets via environment variables
- No `console.log` left in production code — use `logger` (backend) or remove (frontend)

---

## Retell Webhook Security

Always verify Retell webhook signature:
```python
from retell import Retell

retell_client = Retell(api_key=settings.RETELL_API_KEY)

# In webhook handler:
if not retell_client.verify(request_body, headers, settings.RETELL_WEBHOOK_SECRET):
    raise HTTPException(status_code=401, detail="Invalid webhook signature")
```
