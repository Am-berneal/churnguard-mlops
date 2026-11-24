# Google Stitch prompts: ChurnGuard frontend

Use these in [Google Stitch](https://stitch.withgoogle.com) to design the web UI for the churn prediction API.

**How to use them**
1. Upload `docs/brand/logo.png` as an image with **Prompt 0**, so Stitch uses the real logo. Then paste **Prompt 0**. It sets the app's look and its navigation, so every screen matches.
2. Then paste Prompts 1–4 **one at a time** in the same project. Stitch works best with one screen per prompt.
3. Use the **refinement prompts** at the end to fix anything you don't like. Change one thing at a time.
4. Export the code (HTML/Tailwind, or copy it into React) and connect it to the API as shown in the last section.

---

## Prompt 0: App and design system

```
Design a web app called "ChurnGuard", a customer churn prediction tool for a telecom company's retention team. It is the frontend for a machine learning model served by a REST API.

Style: clean, modern SaaS analytics dashboard. Light theme with a white background, slate-gray text, and an indigo primary color (#4F46E5). Risk colors used consistently everywhere: green (#16A34A) for low risk, amber (#F59E0B) for medium risk, red (#DC2626) for high risk. Font: Inter. Rounded corners (12px) on cards, soft shadows, plenty of white space. Desktop first, 1440px wide, but responsive.

Layout: a left sidebar with the attached ChurnGuard logo (an indigo shield outline with a lavender rising line chart inside, and the wordmark "ChurnGuard" with "Guard" in bold). Use the shield icon alone when the sidebar is collapsed and four navigation items with icons: "Overview", "Predict Customer", "Batch Scoring", "Model Insights". At the bottom of the sidebar, a small status pill that says "Model v2 · Live" with a green dot. A top bar with the page title on the left and a "API docs" link button on the right.

Start with the Overview page: a header "Welcome back" with the subtitle "Predict which customers are likely to leave, and act before they do." Below it, a row of four KPI cards: "Model ROC AUC 0.846", "F1 score 0.64", "Decision threshold 0.58", "Model version v2 (champion)". Below that, two large cards side by side: "Predict a single customer" with a short description and an indigo "Start prediction" button, and "Score a CSV file" with a short description and an outlined "Upload CSV" button. At the bottom, a card "Recent predictions" with a table: Customer, Contract, Tenure, Churn probability (shown as a percentage with a small colored bar), Risk (colored pill: Low / Medium / High), Time.
```

## Prompt 1: Predict Customer (form + result)

```
Design the "Predict Customer" page of ChurnGuard, using the same sidebar, colors and style.

Left side (about 60% width): a form card titled "Customer details", split into three sections with small section headings:

Section "Account": Tenure in months (number input with a slider, 0–72), Contract (segmented control: Month-to-month, One year, Two year), Payment method (dropdown: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)), Paperless billing (Yes/No toggle), Monthly charges in $ (number input), Total charges in $ (number input).

Section "Services": Phone service (Yes/No toggle), Multiple lines (dropdown: Yes, No, No phone service), Internet service (segmented control: DSL, Fiber optic, No), then a grid of six dropdowns, each with options Yes / No / No internet service: Online security, Online backup, Device protection, Tech support, Streaming TV, Streaming movies.

Section "Demographics": Gender (Male/Female segmented control), Senior citizen (Yes/No toggle), Partner (Yes/No toggle), Dependents (Yes/No toggle).

At the bottom of the form: a primary indigo button "Predict churn risk" and a text button "Fill sample customer".

Right side (about 40% width), a sticky result card titled "Prediction". Show a large semicircle gauge from 0% to 100% with the needle at 90%, colored red, with the big number "90%" and the label "Churn probability" under it. Below it, a red pill "High risk" and the sentence "This customer is likely to churn." Then a small section "Why this might be" with three bullet points with icons: "Month-to-month contract", "Short tenure (2 months)", "Fiber optic without tech support". Then a section "Suggested actions" with two outlined buttons: "Offer 1-year contract discount" and "Schedule a support call". Also show an empty state version of this card before a prediction is made: an illustration of a shield with the text "Fill in the form to see this customer's churn risk."
```

## Prompt 2: Batch Scoring (CSV upload + results)

```
Design the "Batch Scoring" page of ChurnGuard, same sidebar and style.

At the top, a large dashed-border drag-and-drop upload area with a cloud-upload icon, the text "Drop a CSV of customers here or browse files", and helper text "Columns must match the customer fields. Up to 5,000 rows." Next to it, a small link "Download CSV template".

Below, after an upload, show a results section:
- A row of four summary cards: "Customers scored 1,409", "High risk 312 (22%)" in red, "Medium risk 241 (17%)" in amber, "Low risk 856 (61%)" in green.
- A card with a horizontal stacked bar showing the risk split, and next to it a histogram of churn probabilities from 0 to 1 with a dashed vertical line labeled "Threshold 0.58".
- A results table with a search box, a filter dropdown "Risk: All / High / Medium / Low", and a "Sort by probability" control. Columns: Customer ID, Contract, Tenure, Monthly charges, Churn probability (percentage plus a thin colored bar), Risk (colored pill). Show 10 rows with pagination.
- A primary button "Export results as CSV" at the top right of the table.

Also show a loading state for the upload area: a progress bar with "Scoring 1,409 customers…".
```

## Prompt 3: Model Insights (MLOps page)

```
Design the "Model Insights" page of ChurnGuard, same sidebar and style. This page shows how the machine learning model was built and how good it is, for technical reviewers.

Top: a model header card showing "churn-classifier", a pill "v2 · champion", "Model type: HistGradientBoosting", "Trained: 24 Nov 2026", "Tracked with MLflow", and an outlined button "Open in MLflow" with an external link icon.

Row of six metric cards with small labels: ROC AUC 0.846, PR AUC 0.66, F1 0.64, Precision 0.56, Recall 0.74, Accuracy 0.77. Each card has a tiny info icon with a tooltip explaining the metric in plain English.

A two-column grid of chart cards:
1. "ROC curve": a line chart rising from bottom-left to top-left, with a dashed diagonal baseline and the AUC value in the legend.
2. "Confusion matrix": a 2x2 heatmap with labels Stay / Churn on both axes, in shades of indigo, with counts in each cell.
3. "Top churn drivers": a horizontal bar chart of feature importance: Tenure, Contract, Internet service, Online security, Tech support, Payment method, Multiple lines, Paperless billing.
4. "Model comparison": a grouped bar chart comparing ROC AUC of Logistic Regression 0.842, Random Forest 0.818, Gradient Boosting (baseline) 0.835, Gradient Boosting (tuned) 0.846, with the tuned model highlighted in indigo and labeled "Champion".

At the bottom, a "Version history" timeline: v2 (champion, ROC AUC 0.846, promoted 24 Nov 2026), v1 (archived, ROC AUC 0.846), each with a small "Compare" link.
```

## Prompt 4: Mobile view (optional)

```
Create the mobile version (390px wide) of the ChurnGuard "Predict Customer" page. Replace the sidebar with a bottom tab bar with four icons: Overview, Predict, Batch, Insights. Stack the form sections as collapsible accordions, put the "Predict churn risk" button in a sticky bar at the bottom, and show the result as a bottom sheet that slides up with the gauge, the risk pill and the suggested actions.
```

---

## Refinement prompts (use as needed)

```
Make the gauge on the result card larger and animate the needle from 0 to the prediction value.
```
```
Switch the whole app to a dark theme: background #0F172A, cards #1E293B, keep the same indigo and risk colors.
```
```
Make the form more compact: put two fields per row on desktop and reduce vertical spacing.
```
```
Add a toast notification at the top right that says "Prediction saved" after a prediction is made.
```
```
Add an error state to the result card: a red banner "Can't reach the prediction service. Is the API running on localhost:8000?" with a "Retry" button.
```

---

## Using the logo in the exported code

Copy `docs/brand/` into the frontend's `public/` folder and add these lines to the page `<head>`:

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
```

Use `logo.png` on light backgrounds and `logo-dark.png` on dark ones.

---

## Connecting the exported UI to the API

Start the API (`make serve` or `docker compose up`). CORS already allows `http://localhost:3000` and `http://localhost:5173` (change the `CORS_ORIGINS` env var for other ports).

| Screen | Endpoint | Notes |
|---|---|---|
| Overview KPI cards, Model Insights header & metrics | `GET /model-info` | Returns `version`, `model_type`, `threshold`, `metrics.test_roc_auc`, `metrics.test_f1`, … |
| Sidebar status pill | `GET /health` | `model_loaded: true` → green dot |
| Predict Customer | `POST /predict` | Body = the 19 form fields (names below). Returns `churn_probability`, `will_churn`, `risk_level` |
| Batch Scoring | `POST /predict/batch` | Parse the CSV in the browser and send a JSON array of customers |

The form field names must match the API exactly (case-sensitive):
`gender, SeniorCitizen (0/1), Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

```js
const API = "http://localhost:8000";

export async function predict(customer) {
  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json(); // { churn_probability, will_churn, risk_level }
}

export const modelInfo = () => fetch(`${API}/model-info`).then((r) => r.json());
```

The "Why this might be" bullets and the charts on Model Insights use placeholder data in Stitch. The chart images also exist as MLflow artifacts (`plots/*.png`) in every training run. Exposing them through the API is a good next step for the project.
