"""Sentinel Analyst Console — Streamlit Dashboard."""

import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Sentinel Console", page_icon="🛡️", layout="wide")
st.title("🛡️ Sentinel — Cost-Aware Fraud Detection Console")

API_URL = "http://localhost:8000"
ARTIFACTS = Path("artifacts/sparkov")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Score", "📋 Review Queue", "📊 Cost Dashboard", "📄 Model Card"])

# ============================================================
# TAB 1: Live Score
# ============================================================
with tab1:
    st.subheader("Score a Transaction")
    st.markdown("Enter transaction details or use the sample to see a live decision.")

    col1, col2, col3 = st.columns(3)
    with col1:
        amt = st.number_input("Amount (₹)", min_value=0.01, value=500.0, step=10.0)
        category = st.selectbox("Category", [
            "shopping_net", "misc_net", "grocery_pos", "shopping_pos",
            "gas_transport", "misc_pos", "grocery_net", "travel",
            "entertainment", "personal_care", "kids_pets", "food_dining",
            "home", "health_fitness"
        ])
        merchant = st.text_input("Merchant", "fraud_Kirlin and Sons")
    with col2:
        trans_time = st.text_input("Transaction Time", "2020-06-21 12:14:25")
        cc_num = st.number_input("Card Number", value=2703186189652095, step=1)
        unix_time = st.number_input("Unix Time", value=1371816865, step=1)
    with col3:
        lat = st.number_input("Customer Lat", value=33.9659, format="%.4f")
        long_val = st.number_input("Customer Long", value=-80.9355, format="%.4f")
        merch_lat = st.number_input("Merchant Lat", value=33.986391, format="%.6f")
        merch_long = st.number_input("Merchant Long", value=-81.200714, format="%.6f")

    if st.button("🔍 Score Transaction", type="primary"):
        payload = {
            "trans_date_trans_time": trans_time,
            "cc_num": int(cc_num),
            "merchant": merchant,
            "category": category,
            "amt": float(amt),
            "first": "Test", "last": "User", "gender": "M",
            "street": "123 Test St", "city": "Columbia", "state": "SC",
            "zip": 29209, "lat": lat, "long": long_val,
            "city_pop": 333497, "job": "Engineer",
            "dob": "1985-01-15",
            "trans_num": f"test_{np.random.randint(100000)}",
            "unix_time": int(unix_time),
            "merch_lat": merch_lat, "merch_long": merch_long,
        }
        try:
            resp = requests.post(f"{API_URL}/v1/score", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()

                # Decision banner
                color_map = {"ALLOW": "green", "REVIEW": "orange", "BLOCK": "red"}
                color = color_map.get(data["decision"], "gray")
                st.markdown(
                    f"### Decision: :{color}[**{data['decision']}**] &nbsp; | &nbsp; "
                    f"Risk: **{data['risk_probability']:.4f}** &nbsp; | &nbsp; "
                    f"Latency: **{data['latency_ms']:.1f}ms** &nbsp; | &nbsp; "
                    f"Degraded: **{data['degraded']}**"
                )

                # Cost breakdown
                c1, c2, c3 = st.columns(3)
                c1.metric("₹ if ALLOW", f"₹{data['expected_loss_if_allowed_inr']:,.2f}")
                c2.metric("₹ if REVIEW", f"₹{data['expected_loss_if_reviewed_inr']:,.2f}")
                c3.metric("₹ if BLOCK", f"₹{data['expected_loss_if_blocked_inr']:,.2f}")

                # Reason codes
                if data.get("reason_codes"):
                    st.markdown("#### Reason Codes")
                    for rc in data["reason_codes"]:
                        direction = "🔴" if rc["contribution"] > 0 else "🟢"
                        st.markdown(
                            f"{direction} **{rc['code']}** ({rc['contribution']:+.4f}): {rc['detail']}"
                        )

                # Raw JSON
                with st.expander("Raw API Response"):
                    st.json(data)
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure `uvicorn` is running on port 8000.")

# ============================================================
# TAB 2: Review Queue
# ============================================================
with tab2:
    st.subheader("Review Queue — Transactions Flagged for Manual Review")

    try:
        conn = sqlite3.connect("data/audit.db")
        reviews = pd.read_sql_query(
            "SELECT decision_id, timestamp, amount, risk_probability, decision, model_version, latency_ms, degraded "
            "FROM decisions WHERE decision = 'REVIEW' ORDER BY timestamp DESC LIMIT 50",
            conn,
        )
        conn.close()

        if len(reviews) > 0:
            st.dataframe(
                reviews.style.format({
                    "amount": "₹{:.2f}",
                    "risk_probability": "{:.4f}",
                    "latency_ms": "{:.1f}ms",
                }),
                use_container_width=True,
            )
            st.caption(f"Showing {len(reviews)} most recent REVIEW decisions.")
        else:
            st.info("No transactions in the review queue yet. Score some transactions first.")
    except Exception as e:
        st.error(f"Could not load review queue: {e}")

# ============================================================
# TAB 3: Cost Dashboard
# ============================================================
with tab3:
    st.subheader("Cost Dashboard — Interactive Break-Even Analysis")

    # Interactive cost slider
    st.markdown("#### Adjust Cost Parameters")
    col1, col2 = st.columns(2)
    with col1:
        cb_fee = st.slider("Chargeback Fee (₹)", 500, 3000, 1500, 100)
        margin = st.slider("Gross Margin", 0.05, 0.40, 0.18, 0.01)
    with col2:
        friction = st.slider("Friction Cost (₹)", 50, 500, 250, 25)
        review_cost = st.slider("Review Cost (₹)", 10, 100, 45, 5)

    # Break-even curve with current slider values
    amounts = np.linspace(10, 30000, 300)
    thresholds = []
    for amt_val in amounts:
        A = amt_val * 1.0 + cb_fee  # goods_recovery = 0
        B = margin * amt_val + friction + 0.04 * 6000  # churn_prob * ltv
        p_be = B / (A + B)
        thresholds.append(p_be)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=amounts, y=thresholds, mode="lines",
        name="Break-even threshold", line=dict(color="#3498db", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=amounts, y=[1.0] * len(amounts), fill="tonexty",
        name="BLOCK zone", fillcolor="rgba(231, 76, 60, 0.15)", line=dict(width=0)
    ))
    fig.add_trace(go.Scatter(
        x=amounts, y=[0.0] * len(amounts), fill="tonexty",
        name="ALLOW zone", fillcolor="rgba(46, 204, 113, 0.15)", line=dict(width=0)
    ))
    fig.update_layout(
        title="Break-Even Curve: Threshold as f(Amount)<br><sub>Move the sliders to see how cost parameters shift the curve</sub>",
        xaxis_title="Transaction Amount (₹)",
        yaxis_title="Break-Even Fraud Probability",
        yaxis_range=[0, 1],
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost comparison table
    st.markdown("#### Model Comparison (from evaluation)")
    try:
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
        comparison = pd.DataFrame({
            "Strategy": ["Approve Everything", "Naive 0.5 Threshold", "Sentinel (Cost-Aware)"],
            "Total ₹ Cost": [
                f"₹{metrics.get('total_cost_approve_all_inr', 0):,.0f}",
                f"₹{metrics.get('total_cost_naive_05_inr', 0):,.0f}",
                f"₹{metrics.get('total_cost_sentinel_inr', 0):,.0f}",
            ],
            "Savings": ["—", "87.9%", f"{metrics.get('savings_vs_approve_all_pct', 0):.1f}%"],
        })
        st.table(comparison)
    except Exception:
        st.warning("Metrics file not found.")

    # Audit summary
    st.markdown("#### Decision Distribution (All Time)")
    try:
        conn = sqlite3.connect("data/audit.db")
        dist = pd.read_sql_query(
            "SELECT decision, COUNT(*) as count FROM decisions GROUP BY decision", conn
        )
        conn.close()
        if len(dist) > 0:
            fig2 = go.Figure(data=[go.Bar(
                x=dist["decision"], y=dist["count"],
                marker_color=["#2ecc71", "#f39c12", "#e74c3c"][:len(dist)]
            )])
            fig2.update_layout(title="Decisions Made", height=300)
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

# ============================================================
# TAB 4: Model Card
# ============================================================
with tab4:
    st.subheader("Model Card")

    try:
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())

        st.markdown("#### Model Information")
        st.markdown(f"""
- **Model:** LightGBM gradient boosted trees (500 rounds)
- **Calibration:** Isotonic regression (CalibratedClassifierCV)
- **Training data:** Sparkov fraud detection dataset (1,296,675 transactions)
- **Features:** 15 features across 5 families (temporal, amount, geo, velocity, category)
- **Target:** Binary fraud detection (card-not-present transaction fraud)
        """)

        st.markdown("#### Performance (Held-Out Test Set — Evaluated Once)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        m2.metric("Precision", f"{metrics.get('precision_at_op', 0):.4f}")
        m3.metric("Recall", f"{metrics.get('recall_at_op', 0):.4f}")
        m4.metric("Brier Score", f"{metrics.get('brier_score_calibrated', 0):.6f}")

        st.markdown("#### Cost Impact")
        st.markdown(f"""
- **Approve everything:** ₹{metrics.get('total_cost_approve_all_inr', 0):,.0f}
- **Sentinel (cost-aware):** ₹{metrics.get('total_cost_sentinel_inr', 0):,.0f}
- **Savings:** {metrics.get('savings_vs_approve_all_pct', 0):.1f}%
        """)

        st.markdown("#### Intended Use")
        st.markdown("""
- **Purpose:** Real-time fraud scoring for card-not-present payment transactions
- **Users:** Payment gateway risk teams, merchant fraud analysts
- **Not for:** Credit scoring, identity verification, law enforcement
        """)

        st.markdown("#### Limitations")
        st.markdown("""
- Trained on simulated data (Sparkov) — real-world fraud patterns may differ
- Velocity features require transaction history — new cards have no history
- Cost parameters are estimates and should be tuned per merchant
- No concept drift detection — model should be periodically retrained
        """)

    except Exception as e:
        st.error(f"Could not load model metrics: {e}")