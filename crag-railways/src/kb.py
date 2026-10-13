"""The help desk's knowledge base: handwritten Railways notes in an in-memory Chroma index.

Inputs:  a question string
Outputs: the TOP_K closest passages as Doc objects, with cosine similarity
"""

from dataclasses import dataclass

import chromadb

from src.config import TOP_K

# Simplified notes for the demo. They are not an official IRCTC source.
PASSAGES: dict[str, str] = {
    "tatkal-timing": "Tatkal booking opens one day before the journey date, not counting the journey day. Booking for AC classes opens at 10:00 AM and for non-AC classes at 11:00 AM.",
    "tatkal-aadhaar": "Only IRCTC users whose account is Aadhaar-authenticated can book Tatkal tickets online. An OTP sent to the Aadhaar-linked mobile number is checked at booking.",
    "tatkal-refund": "No refund is given when a confirmed Tatkal ticket is cancelled. A waitlisted Tatkal ticket that is cancelled gets a refund after the usual clerkage charge.",
    "advance-booking": "General reservation opens 60 days before the journey date, not counting the journey day. Foreign tourist quota tickets can be booked 365 days ahead.",
    "cancel-confirmed": "A confirmed ticket cancelled more than 48 hours before departure costs a flat fee per passenger: Rs 240 for AC First Class, Rs 200 for AC 2-tier, Rs 180 for AC 3-tier and AC Chair Car, Rs 120 for Sleeper and Rs 60 for Second Sitting.",
    "cancel-late": "A confirmed ticket cancelled between 48 and 12 hours before departure loses 25 percent of the fare. Between 12 and 4 hours before departure it loses 50 percent. After that, the ticket earns no refund unless a TDR is filed.",
    "waitlist-auto": "An e-ticket that is still fully waitlisted when the chart is prepared is cancelled automatically. The fare minus a clerkage charge goes back to the original payment method.",
    "rac": "An RAC ticket gives a shared side-lower berth. RAC passengers can board the train and are upgraded to a full berth if a confirmed passenger cancels or does not show up.",
    "chart-time": "The first reservation chart is prepared about eight hours before the train departs. After the chart is ready, the waitlist status no longer changes.",
    "tdr": "A TDR (Ticket Deposit Receipt) is filed on the IRCTC website or app for refunds that cancellation rules do not cover, for example when the train runs more than three hours late and the passenger does not travel, or when the AC does not work.",
    "luggage-free": "Each passenger may carry free luggage of 70 kg in AC First Class, 50 kg in AC 2-tier, 40 kg in AC 3-tier, AC Chair Car and Sleeper, and 35 kg in Second Sitting.",
    "luggage-extra": "Luggage above the free allowance must be booked at the parcel office at least 30 minutes before departure. Unbooked excess luggage found on board is charged at six times the luggage rate.",
    "child-fare": "Children under five travel free without a separate berth. Children aged five to eleven pay full fare for a berth, or half fare if the parents choose no berth.",
    "senior-concession": "The senior citizen fare concession was withdrawn in March 2020 and has not been brought back. Senior citizens can still request a lower berth at booking.",
    "boarding-change": "The boarding station on a ticket can be changed online up to four hours before the chart is prepared. After that change, the passenger cannot board from the original station.",
}


@dataclass
class Doc:
    """One retrieved passage and how close it was to the question."""

    id: str
    text: str
    similarity: float


def build_collection() -> chromadb.Collection:
    """Embed every passage into a fresh in-memory collection that uses cosine distance."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        "railways", configuration={"hnsw": {"space": "cosine"}}
    )
    if collection.count() == 0:
        collection.add(ids=list(PASSAGES), documents=list(PASSAGES.values()))
    return collection


_collection: chromadb.Collection | None = None


def retrieve(question: str, k: int = TOP_K) -> list[Doc]:
    """Return the k passages closest to the question, best first."""
    global _collection
    if _collection is None:
        _collection = build_collection()
    result = _collection.query(query_texts=[question], n_results=k)
    ids, texts, distances = result["ids"][0], result["documents"][0], result["distances"][0]
    # Chroma returns a cosine DISTANCE; similarity is 1 minus it.
    return [Doc(i, t, round(1 - d, 2)) for i, t, d in zip(ids, texts, distances)]


if __name__ == "__main__":
    for q in [
        "What time does Tatkal booking open for AC classes?",
        "How much check-in baggage can I carry on an IndiGo domestic flight?",
        "My train ran 5 hours late and I didn't travel. How do I get my money back and how many days does the refund take?",
    ]:
        print(f"\nQ: {q}")
        for doc in retrieve(q):
            print(f"  {doc.similarity:.2f}  {doc.id}")
