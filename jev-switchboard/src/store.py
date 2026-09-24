"""The existing store backend: orders, catalog, return policy and two plain tools.

Inputs:  a customer message (for order-ID extraction).
Outputs: order facts and computed numbers. No AI anywhere in this file.
"""

import re
from datetime import date

TODAY = date(2026, 9, 23)  # fixed so the demo gives the same answers every run
RETURN_WINDOW_DAYS = 7

ORDERS: dict[str, dict] = {
    "KH-1042": {"item": "Jaipur block-print bedsheet", "price": 1499, "status": "shipped",
                "courier": "Delhivery", "eta": "2026-09-25", "delivered_on": None},
    "KH-1043": {"item": "Brass diya set (4 pcs)", "price": 899, "status": "delivered",
                "courier": "Blue Dart", "eta": None, "delivered_on": "2026-09-18"},
    "KH-1044": {"item": "Handloom cotton saree", "price": 3299, "status": "delivered",
                "courier": "Ekart", "eta": None, "delivered_on": "2026-08-30"},
}

CATALOG = """\
- Jaipur block-print bedsheet, king size, 100% cotton, Rs 1,499. Machine wash cold.
- Brass diya set (4 pcs), hand-polished, Rs 899. Wipe with a dry cloth, no soap.
- Handloom cotton saree, 5.5 m with blouse piece, Rs 3,299. Dry clean first wash.
- Terracotta planter, 8 inch, drainage hole, Rs 649. Not frost-proof."""

RETURN_POLICY = f"""\
- Returns and exchanges are accepted within {RETURN_WINDOW_DAYS} days of delivery.
- Items must be unused, with tags. Sarees must be unwashed.
- Refunds go back to the original payment method within 5-7 working days.
- No discounts, coupons or compensation are offered for returns or delays."""

ORDER_ID = re.compile(r"\bKH-\d{4}\b", re.IGNORECASE)


def find_order_id(message: str) -> str | None:
    """Pull the first order ID (like KH-1042) out of a message, or None."""
    match = ORDER_ID.search(message)
    return match.group(0).upper() if match else None


def order_lookup(message: str) -> str:
    """Answer an order-status question straight from the database."""
    order_id = find_order_id(message)
    if order_id is None:
        return "Please share your order ID (it looks like KH-1234) and I'll check it."
    order = ORDERS.get(order_id)
    if order is None:
        return f"I couldn't find order {order_id}. Please double-check the ID."
    if order["status"] == "delivered":
        return f"{order_id} ({order['item']}) was delivered on {order['delivered_on']}."
    return (f"{order_id} ({order['item']}) is {order['status']} with {order['courier']}, "
            f"expected by {order['eta']}.")


def days_since_delivery(order_id: str | None) -> int | None:
    """Days between delivery and today. Date math lives in code, never in a model."""
    order = ORDERS.get(order_id or "")
    if order is None or order["delivered_on"] is None:
        return None
    return (TODAY - date.fromisoformat(order["delivered_on"])).days
