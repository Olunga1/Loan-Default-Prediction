"use client";

import { useState } from "react";
import { explainPrediction, predictLoan, type ExplanationResponse, type LoanApplicationRequest, type PredictionResponse } from "@/lib/api";

const initialForm: LoanApplicationRequest = {
  loan_id: "LN_DEMO_001",
  age: 35,
  income: 85000,
  loan_amount: 250000,
  credit_score: 720,
  months_employed: 72,
  num_credit_lines: 3,
  interest_rate: 4.5,
  loan_term: 360,
  dti_ratio: 0.35,
  education: "Bachelor's",
  employment_type: "Full-time",
  marital_status: "Married",
  has_mortgage: "Yes",
  has_dependents: "Yes",
  loan_purpose: "Home",
  has_cosigner: "No",
};

export default function Home() {
  const [form, setForm] = useState<LoanApplicationRequest>(initialForm);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const pred = await predictLoan(form);
      setPrediction(pred);
      const expl = await explainPrediction(form);
      setExplanation(expl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function setField<K extends keyof LoanApplicationRequest>(key: K, value: LoanApplicationRequest[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <main style={{ fontFamily: "Arial, sans-serif", padding: 24, maxWidth: 960, margin: "0 auto" }}>
      <h1>Loan Default Risk Assessment</h1>
      <p>Minimal working frontend baseline.</p>

      <form onSubmit={onSubmit} style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
        <label>Loan ID<input value={form.loan_id} onChange={(e) => setField("loan_id", e.target.value)} /></label>
        <label>Age<input type="number" value={form.age} onChange={(e) => setField("age", Number(e.target.value))} /></label>
        <label>Income<input type="number" value={form.income} onChange={(e) => setField("income", Number(e.target.value))} /></label>
        <label>Loan Amount<input type="number" value={form.loan_amount} onChange={(e) => setField("loan_amount", Number(e.target.value))} /></label>
        <label>Credit Score<input type="number" value={form.credit_score} onChange={(e) => setField("credit_score", Number(e.target.value))} /></label>
        <label>DTI Ratio<input type="number" step="0.01" value={form.dti_ratio} onChange={(e) => setField("dti_ratio", Number(e.target.value))} /></label>
        <button type="submit" disabled={loading} style={{ gridColumn: "1 / -1", padding: "10px 14px" }}>
          {loading ? "Predicting..." : "Run Prediction"}
        </button>
      </form>

      {error ? <p style={{ color: "crimson" }}>{error}</p> : null}

      {prediction ? (
        <section style={{ marginTop: 20 }}>
          <h2>Prediction</h2>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </section>
      ) : null}

      {explanation ? (
        <section>
          <h2>Top Feature Contributions</h2>
          <pre>{JSON.stringify(Object.entries(explanation.feature_contributions).slice(0, 10), null, 2)}</pre>
        </section>
      ) : null}
    </main>
  );
}
