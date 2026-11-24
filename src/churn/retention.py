"""Turn a churn score into a business decision.

- risk level and priority (who to call first)
- revenue at risk (how much the customer is worth keeping)
- recommended actions from a retention playbook, matched to the reasons
"""
from dataclasses import asdict, dataclass

# reason feature -> (internal action, customer-facing offer, channel, cost)
_PLAYS = {
    "Contract": ("Offer a discounted 1-year contract",
                 "a discounted 1-year plan", "Phone call", "medium"),
    "tenure": ("Enroll in the new-customer onboarding program",
               "a free setup session with our team to get the most out of your plan",
               "Email + call", "low"),
    "TechSupport": ("Add 3 months of free tech support",
                    "3 months of free tech support", "In-app offer", "low"),
    "OnlineSecurity": ("Add 3 months of free online security",
                       "3 months of free online security", "In-app offer", "low"),
    "OnlineBackup": ("Offer a free online backup trial",
                     "a free online backup trial", "Email", "low"),
    "DeviceProtection": ("Offer a device protection trial",
                         "a free device protection trial", "Email", "low"),
    "PaymentMethod": ("Give a small credit for switching to auto-pay",
                      "a bill credit when you switch to automatic payments", "Email", "low"),
    "MonthlyCharges": ("Review the plan and suggest a better-value bundle",
                       "a quick plan review to find you a better-value bundle",
                       "Phone call", "medium"),
    "InternetService": ("Run a proactive service-quality check",
                        "a free check of your internet connection quality",
                        "Technician follow-up", "medium"),
    "PaperlessBilling": ("Send a clear monthly bill summary",
                         "a clearer monthly bill summary", "Email", "low"),
    "SeniorCitizen": ("Offer the senior support line and a simplified plan",
                      "our dedicated senior support line and a simpler plan", "Phone call", "low"),
}
PLAYBOOK: dict[str, dict] = {
    k: dict(zip(("action", "offer", "channel", "cost"), v, strict=True)) for k, v in _PLAYS.items()
}
DEFAULT_ACTION = {"action": "Send a loyalty thank-you offer", "offer": "a loyalty thank-you reward",
                  "channel": "Email", "cost": "low"}


@dataclass
class Decision:
    risk_level: str
    priority: str
    revenue_at_risk: float
    actions: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def risk_level(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.7):
        return "high"
    if p >= threshold * 0.6:
        return "medium"
    return "low"


def revenue_at_risk(p: float, monthly_charges: float, months: int = 12) -> float:
    """Expected revenue lost over the next ``months`` if we do nothing."""
    return round(float(p) * float(monthly_charges) * months, 2)


def priority(level: str, p: float, revenue: float, threshold: float) -> str:
    """P1 = high risk and valuable (call today), P2 = predicted to churn, P3 = watch list."""
    if level == "high" and revenue >= 600:
        return "P1"
    if p >= threshold:
        return "P2"
    if level == "medium":
        return "P3"
    return "—"


def recommend(reasons: list[dict], level: str, max_actions: int = 2) -> list[dict]:
    if level == "low":
        return []
    actions, seen = [], set()
    for r in reasons:
        play = PLAYBOOK.get(r["feature"])
        if play and play["action"] not in seen:
            actions.append({**play, "because": r["reason"]})
            seen.add(play["action"])
        if len(actions) == max_actions:
            break
    return actions or [{**DEFAULT_ACTION, "because": "General churn risk"}]


def decide(p: float, monthly_charges: float, reasons: list[dict], threshold: float) -> Decision:
    level = risk_level(p, threshold)
    revenue = revenue_at_risk(p, monthly_charges)
    return Decision(level, priority(level, p, revenue, threshold), revenue, recommend(reasons, level))
