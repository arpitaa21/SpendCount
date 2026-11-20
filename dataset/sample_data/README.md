# SpendCount — synthetic student transaction data

> **Synthetic demo data only.** Every record is fictional and was generated for software development and QA. No real person's financial information is included. Amounts are illustrative INR values.

## Dataset files

- `student_transactions_6_months.csv` — six months (1 May–31 October 2026) of student-style activity, including recurring pocket money, hostel/mess charges, mobile recharge, subscriptions, campus food, transport, occasional food delivery, academic purchases, freelance income, scholarship and family support. Includes plausible one-off costs and month-to-month variation for dashboard and forecasting tests.
- `anomaly_detection_test.csv` — compact QA fixture with normal baseline transactions, a high-value repair, an unusually large online purchase and a repeated same-day payment. The planted cases are test fixtures, not evidence of real-world anomaly rates.
- `invalid_transactions.csv` — upload-validation fixture containing malformed dates, non-numeric and negative amounts, missing required values and an unsupported transaction type.

## CSV schema

| Column | Description | Example |
|---|---|---|
| `transaction_date` | Transaction date in ISO format (`YYYY-MM-DD`) | `2026-05-02` |
| `description` | Human-readable transaction label | `Monthly Pocket Money` |
| `amount` | Positive numeric amount in INR | `6500.00` |
| `transaction_type` | `Income` or `Expense` | `Expense` |
| `category` | Transaction category | `Food`, `Hostel`, `Freelance` |

Income categories include Pocket Money, Scholarship, Freelance and Family Support. Expense categories include Food, Transport, Hostel, Shopping, Subscriptions, Academics, Entertainment and Utilities.

## Suggested import validation

1. Require all five columns; column order may be normalized.
2. Parse real calendar dates strictly.
3. Require non-empty descriptions and categories.
4. Require finite positive amounts. Transaction direction is represented by `transaction_type`.
5. Accept only `Income` and `Expense` (case-insensitive if supported).
6. Report invalid row numbers and actionable error messages.

## Product and model testing notes

- Keep a visible **Synthetic Demo Data** label wherever these records appear in the UI.
- Do not automatically seed production accounts with this data.
- Compute totals, category breakdowns, forecasts and AI insights from the active user's imported or manually entered transactions. Never hardcode figures or insights from these fixtures.
- Verify the same workflows with real-user CSV uploads and manual entry.
- These synthetic records support functional testing only. They must not be presented as proof of real-world forecasting or anomaly-detection accuracy.
