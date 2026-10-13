"""The store policy and a day of customer questions.

Inputs: none (all data is inline).
Outputs: POLICY (the text the agent answers from) and TRAFFIC, a list of
(question, intent) pairs in arrival order. The intent tag is only used to
grade cache hits; the agent never sees it.
"""

POLICY = """Kumkum Home - customer support policy (version {version})

Returns: unused items can be returned within {return_days} days of delivery.
Custom-stitched items cannot be returned.
Refunds: prepaid orders are refunded to the original payment method within 5-7 working days
of the return reaching our warehouse. COD orders are refunded by bank transfer (NEFT) within
7-10 working days, after you share your bank details.
Cancellation: an order can be cancelled free of charge until it is dispatched.
After dispatch, refuse the parcel at the door instead.
Subscriptions: the monthly candle box subscription can be cancelled anytime from
My Account > Subscriptions; the current month's box still ships.
Cash on delivery: available for orders up to Rs 5,000. A Rs 49 COD fee applies.
Delivery: metro cities in 2-4 days, other locations in 5-7 days. Free shipping above Rs 999.
Warranty: brass and copper items carry a 1-year warranty against manufacturing defects.
Exchange: size or colour exchange is free, once per order, within the return window.
Damaged items: report damage within 48 hours of delivery with a photo of the parcel.
Gift wrap: available for Rs 50 per item at checkout.
Contact: support@kumkumhome.in, 10am-7pm IST, Monday to Saturday.
"""

# (question, intent). Many intents come back as paraphrases, the way real
# support traffic does. Some pairs sound alike but need different answers:
# cancel an order vs a subscription, prepaid vs COD refunds, COD limit vs fee.
TRAFFIC: list[tuple[str, str]] = [
    ("How many days do I have to return an item?", "returns_window"),
    ("Can I cancel my order?", "cancel_order"),
    ("What is the return window?", "returns_window"),
    ("How long does delivery take?", "delivery_time"),
    ("Is cash on delivery available?", "cod_limit"),
    ("When will I get my refund for a prepaid order?", "refund_prepaid"),
    ("how many days to return a product", "returns_window"),
    ("How do I cancel my candle subscription?", "cancel_subscription"),
    ("Is there a warranty on brass items?", "warranty"),
    ("I want to cancel my order, how?", "cancel_order"),
    ("How long will delivery take to my city?", "delivery_time"),
    ("When do I get my refund for a COD order?", "refund_cod"),
    ("Can I return something after a week?", "returns_window"),
    ("Do you charge extra for cash on delivery?", "cod_fee"),
    ("Is shipping free?", "free_shipping"),
    ("My item arrived damaged, what do I do?", "damaged"),
    ("can i cancel the order", "cancel_order"),
    ("Can I exchange for a different size?", "exchange"),
    ("What is the return window for items?", "returns_window"),
    ("Do brass products come with a warranty?", "warranty"),
    ("How long does a prepaid refund take?", "refund_prepaid"),
    ("Do you offer gift wrapping?", "gift_wrap"),
    ("How do I cancel my order?", "cancel_order"),
    ("How can I contact customer support?", "contact"),
    ("How many days for delivery?", "delivery_time"),
    ("What is the maximum order value for cash on delivery?", "cod_limit"),
    ("How many days do I get to return an item?", "returns_window"),
    ("The product came broken. What should I do?", "damaged"),
    ("How do I stop my subscription?", "cancel_subscription"),
    ("What is the delivery time?", "delivery_time"),
    ("Can I return a custom stitched item?", "returns_custom"),
    ("Is there a fee for COD?", "cod_fee"),
    ("I want to cancel my order", "cancel_order"),
    ("Is shipping free on my order?", "free_shipping"),
    ("How long does it take to refund a COD order?", "refund_cod"),
    ("What's your return window?", "returns_window"),
    ("Can I exchange an item for a different colour?", "exchange"),
    ("How long will delivery take?", "delivery_time"),
    ("Is there a warranty on copper items?", "warranty"),
    ("Can I cancel my subscription?", "cancel_subscription"),
    ("When will I receive my refund for a prepaid order?", "refund_prepaid"),
    ("What are your customer support hours?", "contact"),
    ("Can I cancel my order?", "cancel_order"),
    ("Item arrived damaged, what should I do?", "damaged"),
    ("Do you have gift wrap?", "gift_wrap"),
    ("How many days do I have to return an item?", "returns_window"),
    ("Is cash on delivery available for my order?", "cod_limit"),
    ("Delivery time to a small town?", "delivery_time"),
    ("Is there any extra charge for cash on delivery?", "cod_fee"),
    ("When will my refund come for a COD order?", "refund_cod"),
    ("Can I exchange it for another size?", "exchange"),
    ("Is shipping free?", "free_shipping"),
    ("Can I return an item after a week?", "returns_window"),
    ("How do I reach customer support?", "contact"),
    ("How can I cancel my order?", "cancel_order"),
    ("Do brass items have a warranty?", "warranty"),
    ("How long does delivery take?", "delivery_time"),
    ("Can I return custom stitched items?", "returns_custom"),
    ("What is the return window?", "returns_window"),
    ("How long does a prepaid refund take?", "refund_prepaid"),
]
