"""Optional Calendar + Gmail tools. build_google_tools() returns an empty list if
the reader hasn't done the Google Cloud OAuth setup (no credentials.json) or
hasn't installed the optional Google packages — the bot runs fine either way.

Inputs:  config.CREDENTIALS_PATH (OAuth client secret) / config.TOKEN_PATH (cache).
Outputs: build_google_tools() -> list[Tool], possibly empty.

Setup (only if you want this): uv add google-api-python-client google-auth-httplib2
google-auth-oauthlib, then download an OAuth client secret from Google Cloud
Console and save it as credentials.json in this project's root.
"""

import base64
from datetime import datetime, timezone
from email.mime.text import MIMEText

from langchain_core.tools import tool

from . import config

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def _now_rfc3339() -> str:
    """Current UTC time in the RFC3339 format the Calendar API's timeMin expects."""
    return datetime.now(timezone.utc).isoformat()


def _authorize():
    """Load a cached OAuth token, refreshing or running the interactive flow once."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if config.TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(config.TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(config.CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        config.TOKEN_PATH.write_text(creds.to_json())
    return creds


def build_google_tools() -> list:
    """Build Calendar + Gmail tools, or [] if setup is incomplete."""
    if not config.CREDENTIALS_PATH.exists():
        return []
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return []

    creds = _authorize()
    calendar = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)

    @tool
    def calendar_read(max_results: int = 5) -> str:
        """List the next upcoming Google Calendar events."""
        events = calendar.events().list(
            calendarId="primary", maxResults=max_results, singleEvents=True,
            orderBy="startTime", timeMin=_now_rfc3339(),
        ).execute().get("items", [])
        if not events:
            return "No upcoming events."
        lines = []
        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date"))
            lines.append(f"- {start}: {e.get('summary', '(no title)')}")
        return "\n".join(lines)

    @tool
    def calendar_create(summary: str, start_iso: str, end_iso: str) -> str:
        """Create a Google Calendar event. start_iso/end_iso are full ISO 8601
        datetimes with timezone — compute them from current_datetime, never guess."""
        event = calendar.events().insert(calendarId="primary", body={
            "summary": summary,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
        }).execute()
        return f"Created event: {event.get('htmlLink', summary)}"

    @tool
    def email_read(max_results: int = 5) -> str:
        """List the sender and subject of the most recent Gmail messages."""
        msgs = gmail.users().messages().list(
            userId="me", maxResults=max_results
        ).execute().get("messages", [])
        if not msgs:
            return "No recent messages."
        lines = []
        for m in msgs:
            full = gmail.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()
            headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
            lines.append(f"- From {headers.get('From', '?')}: {headers.get('Subject', '(no subject)')}")
        return "\n".join(lines)

    @tool
    def email_draft(to: str, subject: str, body: str) -> str:
        """Create a Gmail draft. This never sends email — it only saves a draft
        for the user to review and send themselves from Gmail."""
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        return f'Draft saved to {to}: "{subject}". Not sent — review it in Gmail.'

    return [calendar_read, calendar_create, email_read, email_draft]
