"""Personalized retention messages, written by Gemini when available.

Set ``GEMINI_API_KEY`` (free key from https://aistudio.google.com/apikey) to
have Gemini write the message. Without a key, or if the call fails, a
template-based message is returned, so the feature always works offline.

Only model outputs (risk, reasons, offer) are sent, never names or IDs.
"""
import json
import os
import urllib.request

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def build_prompt(customer: dict, reasons: list[dict], actions: list[dict], tone: str) -> str:
    reason_text = "; ".join(r["reason"] for r in reasons) or "no single dominant reason"
    offer = actions[0].get("offer", actions[0]["action"]) if actions else "a loyalty thank-you reward"
    return (
        "You are a customer retention specialist at a telecom company.\n"
        f"Write a short {tone} message (max 90 words) to an existing customer.\n"
        f"Customer facts: {customer.get('tenure')} months with us, "
        f"{customer.get('Contract')} contract, {customer.get('InternetService')} internet, "
        f"pays ${customer.get('MonthlyCharges')} per month.\n"
        f"Our churn model's main risk factors for them: {reason_text}.\n"
        f"The offer to make: {offer}.\n"
        "Rules: do not mention churn, risk, models or data analysis. Do not invent prices "
        "or discounts that were not given. Sound human and warm. End with one clear next step. "
        "Return only the message text, starting with 'Hi there,'."
    )


def template_message(customer: dict, actions: list[dict]) -> str:
    months = int(customer.get("tenure", 0))
    offer = actions[0].get("offer", actions[0]["action"]) if actions else "a loyalty thank-you reward"
    opener = ("Welcome aboard — we're glad you chose us." if months < 6
              else f"Thank you for being with us for {months} months.")
    return (f"Hi there,\n\n{opener} We'd love to make sure your plan still works for you, "
            f"so we'd like to offer you {offer}. It only takes a couple of minutes to set up.\n\n"
            "Just reply to this message or call us, and we'll take care of the rest.\n\n"
            "— The Customer Care Team")


def _call_gemini(prompt: str, api_key: str, timeout: int = 20) -> str:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
    }).encode()
    req = urllib.request.Request(
        GEMINI_URL.format(model=GEMINI_MODEL), data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        data = json.load(resp)
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def retention_message(customer: dict, reasons: list[dict], actions: list[dict],
                      tone: str = "friendly") -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            text = _call_gemini(build_prompt(customer, reasons, actions, tone), api_key)
            return {"message": text, "source": f"gemini:{GEMINI_MODEL}"}
        except Exception as e:  # noqa: BLE001 - network/quota errors fall back to template
            return {"message": template_message(customer, actions), "source": "template",
                    "note": f"Gemini unavailable: {type(e).__name__}"}
    return {"message": template_message(customer, actions), "source": "template"}
