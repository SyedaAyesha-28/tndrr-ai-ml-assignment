"""
Task 1 data generator: customer complaint triage.
Input : a customer complaint message
Output: {"category": ..., "urgency": ..., "sentiment": ...}

Run:  python make_data.py
Makes: train.jsonl (1500), test.jsonl (100)

Idea: labels are CHOSEN first, then the message is BUILT to match them,
so labels are always consistent. The last core template of every category
is used ONLY in the test set, so the test set has unseen phrasings.
"""
import json
import random
from collections import Counter

random.seed(42)

APPS = ["Swiggy", "Zomato", "Amazon", "Flipkart", "Myntra", "Blinkit", "PhonePe", "Paytm", "BigBasket"]
ITEMS = ["phone", "shoes", "mixer grinder", "laptop bag", "headphones", "t-shirt", "water bottle",
         "biryani", "pizza", "groceries", "smartwatch", "kurta", "books", "charger"]

# category -> core templates (last one is test-only)
CORE = {
    "refund": [
        "my refund for the {item} order on {app} has not come yet",
        "I returned the {item} but the money is not back in my account",
        "{app} said refund of Rs {amt} will be processed but nothing received",
        "still waiting for my refund of Rs {amt}",
        "paisa wapas nahi aaya for the cancelled {item} order",  # test-only
    ],
    "delivery_delay": [
        "my {item} order from {app} has not been delivered yet",
        "the delivery is very late, tracking shows no update",
        "my order was supposed to arrive already but it is stuck",
        "{app} order not delivered, the app keeps saying out for delivery",
        "mera {item} abhi tak nahi pahuncha",  # test-only
    ],
    "wrong_item": [
        "I ordered a {item} but received something completely different",
        "{app} sent me the wrong {item}",
        "the package had a different product than what I ordered",
        "got the wrong size of {item}, not what I selected",
        "box mein {item} ki jagah kuch aur nikla",  # test-only
    ],
    "damaged_product": [
        "the {item} arrived broken",
        "my {item} from {app} was damaged when it reached me",
        "the packaging was torn and the {item} is defective",
        "{item} is not working, it was dead on arrival",
        "the {item} had cracks all over when I opened the parcel",  # test-only
    ],
    "payment_failed": [
        "money got deducted from my account but the {app} order was not placed",
        "payment of Rs {amt} failed but I was charged",
        "I was charged twice for the same {item} order",
        "my UPI payment went through but {app} shows unpaid",
        "Rs {amt} cut gaya but order confirm nahi hua",  # test-only
    ],
    "account_issue": [
        "I cannot log in to my {app} account",
        "my {app} account got blocked without any reason",
        "I am not receiving the OTP to sign in",
        "I want to change my registered phone number but the app does not allow it",
        "my account shows someone else's order history",  # test-only
    ],
    "poor_service": [
        "the customer care executive was rude to me",
        "{app} support keeps giving copy paste replies and no solution",
        "nobody from support is picking up my calls",
        "the delivery guy behaved badly with me",
        "support closed my ticket without solving anything",  # test-only
    ],
}

# urgency cues
URGENCY = {
    "high": [
        "it has been {days} days now",
        "this is the third time I am writing about this",
        "I need this sorted today, it is urgent",
        "I have been waiting for {days} days",
        "this is a big amount for me, please act immediately",
    ],
    "medium": [
        "it has been {days} days",
        "please look into this soon",
        "I have waited a few days already",
    ],
    "low": [
        "no rush, whenever you get time",
        "just reporting it, it is just placed yesterday",
        "it happened only today",
        "not urgent but please check",
    ],
}

# sentiment: (prefix, suffix)
SENTIMENT = {
    "angry": [
        ("", "This is unacceptable!!"),
        ("", "Worst service ever, I am done with you people!"),
        ("What nonsense is this?? ", "Fix it NOW."),
    ],
    "frustrated": [
        ("", "This is really frustrating."),
        ("Honestly I am quite disappointed. ", ""),
        ("", "I expected better from you."),
    ],
    "neutral": [
        ("", "Please advise."),
        ("", ""),
        ("Reporting an issue: ", ""),
    ],
    "polite": [
        ("Hi team, ", "Thanks in advance for your help."),
        ("Hello, ", "Would really appreciate your help, thank you."),
        ("Good morning, ", "Kindly check when you can."),
    ],
}

CATS = list(CORE)
URG_DAYS = {"high": (7, 30), "medium": (3, 6), "low": (1, 2)}


def build(test_mode: bool):
    cat = random.choice(CATS)
    urg = random.choices(["high", "medium", "low"], weights=[0.35, 0.35, 0.30])[0]
    sen = random.choice(list(SENTIMENT))

    templates = CORE[cat]
    core_t = templates[-1] if (test_mode and random.random() < 0.4) else random.choice(templates[:-1])
    lo, hi = URG_DAYS[urg]
    fill = dict(app=random.choice(APPS), item=random.choice(ITEMS),
                amt=random.choice([199, 450, 799, 1200, 2499, 5000]), days=random.randint(lo, hi))

    core = core_t.format(**fill)
    cue = random.choice(URGENCY[urg]).format(**fill)
    pre, suf = random.choice(SENTIMENT[sen])

    body = f"{core}, {cue}." if random.random() < 0.5 else f"{cue}, {core}."
    if pre.endswith(": ") or pre.endswith(", "):
        body = body[0].lower() + body[1:] if pre.endswith(", ") else body
    else:
        body = body[0].upper() + body[1:]
    text = " ".join(x for x in [pre.strip(), body, suf] if x)
    text = text.replace(" ,", ",").replace("  ", " ").strip()

    label = {"category": cat, "urgency": urg, "sentiment": sen}
    return text, label


def make(n, test_mode):
    seen, rows = set(), []
    while len(rows) < n:
        text, label = build(test_mode)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"input": text, "output": json.dumps(label)})
    return rows


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    train = make(1500, test_mode=False)
    test = make(100, test_mode=True)
    train_texts = {r["input"] for r in train}
    test = [r for r in test if r["input"] not in train_texts]  # no leakage
    save("train.jsonl", train)
    save("test.jsonl", test)
    print(f"train={len(train)} test={len(test)}")
    for field in ["category", "urgency", "sentiment"]:
        c = Counter(json.loads(r["output"])[field] for r in test)
        print(field, dict(c))
    print("\nSample:")
    for r in train[:3]:
        print(r["input"], "->", r["output"])
