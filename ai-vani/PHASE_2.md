# Phase 2 — Phone Call In

**Goal:** Real phone number → Twilio SIP → LiveKit → agent picks up.
**Showable:** Call number → hear agent greeting.
**Cost:** Twilio free trial ($15 credit — CONFIRM WITH USER before setup)

---

## Before writing any code — check these docs
1. https://docs.livekit.io/sip/ (LiveKit SIP overview)
2. https://docs.livekit.io/sip/quickstart/ (SIP quickstart)
3. Twilio SIP trunk setup (search for latest guide)

## Step-by-step instructions

### Step 1 — Confirm Twilio cost with user
ALWAYS tell user: "Twilio has a free trial with $15 credit. Inbound calls cost ~$0.0085/min.
Do you want to proceed with Twilio or do you have an existing Twilio account?"

### Step 2 — LiveKit SIP configuration
Instruct user to configure LiveKit SIP inbound trunk.
Check latest docs at https://docs.livekit.io/sip/ before writing any config.

### Step 3 — Twilio SIP trunk
Instruct user to create Twilio SIP trunk pointing to LiveKit SIP URI.
Search for latest Twilio + LiveKit SIP guide before giving instructions.

### Step 4 — Agent dispatch rule
Write LiveKit dispatch rule so inbound SIP calls trigger the agent.

### Step 5 — Inbound call handler
Update agent.py to handle inbound call context (caller number, etc).

## Acceptance criteria
- [ ] Calling the number triggers the agent
- [ ] Agent picks up within 3 seconds
- [ ] Agent greets caller audibly
- [ ] Call can be ended cleanly

