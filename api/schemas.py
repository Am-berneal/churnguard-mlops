"""Request and response models for the ChurnGuard API."""
from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "customerID": "DEMO-0001",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.1,
}


class Customer(BaseModel):
    customerID: str | None = Field(None, description="Optional; echoed back, never used by the model")
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": EXAMPLE_CUSTOMER}}


class Reason(BaseModel):
    feature: str
    reason: str
    impact: float = Field(description="Drop in churn probability if this factor matched a loyal customer")


class Action(BaseModel):
    action: str
    offer: str
    channel: str
    cost: str
    because: str


class Prediction(BaseModel):
    customerID: str | None = None
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]
    priority: str
    revenue_at_risk: float = Field(description="Expected revenue lost over 12 months if nothing is done")
    reasons: list[Reason]
    actions: list[Action]


class BatchSummary(BaseModel):
    n_customers: int
    n_high: int
    n_medium: int
    n_low: int
    n_p1: int
    revenue_at_risk_total: float
    avg_churn_probability: float


class BatchResult(BaseModel):
    summary: BatchSummary
    results: list[Prediction]


class MessageRequest(BaseModel):
    customer: Customer
    tone: Literal["friendly", "professional", "concise"] = "friendly"


class MessageResponse(BaseModel):
    message: str
    source: str
    note: str | None = None
