# Phase 4 — Real Booking Conversation

**Goal:** Full conversation flow: book / cancel / reschedule via pluggable booking backend.
**Showable:** Full call from greeting to booking confirmed — no human needed.
**Cost:** FREE in dev (uses mock backend).

---

## Before writing any code — check these docs
1. Get today's date first: run `date +"%B %d, %Y"`
2. Check latest LangGraph: https://langchain-ai.github.io/langgraph/
3. Check latest version: https://pypi.org/project/langgraph/

## Booking Intents to handle
1. Book new appointment
2. Reschedule existing appointment
3. Cancel appointment
4. General questions (hours, location, services)
5. Wrong number / hang up

## Booking API interface (generic — business plugs in their own)
- GET  /slots?date_from=...&date_to=...&provider_id=...
- POST /bookings
- PUT  /bookings/{id}
- DELETE /bookings/{id}
- GET  /bookings?customer_name=...

## Step-by-step instructions

### Step 1 — Mock booking backend
Build src/booking/mock.py first — in-memory bookings, no real DB.
This lets Phase 4 work completely offline for testing.

### Step 2 — LangGraph graph design
Before coding, map the conversation nodes:
- Greeting → Intent detection
- Book: collect name → pick slot → confirm → call API
- Cancel: collect name → find booking → confirm → call API
- Reschedule: collect name → find booking → pick new slot → confirm → call API
- Fallback: unknown intent → offer to repeat or transfer

### Step 3 — Tool functions
Async tool functions wrapping the booking API client.
Each tool handles API failures with a user-facing fallback message.

### Step 4 — Wire LangGraph into LiveKit agent
Replace echo logic with LangGraph agent.
Session state (customer name, booking id, intent) lives per call, cleaned up on disconnect.

### Step 5 — End call
On LangGraph terminal node → trigger room disconnect via LiveKit API.

## Acceptance criteria
- [ ] Book appointment end-to-end (mock backend)
- [ ] Cancel appointment end-to-end
- [ ] Reschedule end-to-end
- [ ] API failure → agent says helpful message, no crash
- [ ] Call ends cleanly after task complete
- [ ] Swap mock for real backend by changing one env var

