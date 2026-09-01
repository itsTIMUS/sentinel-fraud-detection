"""Sentinel Analyst Console — Production Dashboard."""

import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# --- Page config ---
st.set_page_config(
    page_title="Sentinel • Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for professional look ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a4a;
    }
    .main-header h1 { color: #00d4ff; font-size: 2rem; margin: 0; }
    .main-header p { color: #8892b0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
    .metric-card .label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
    .decision-allow { background: #0d4d2e; color: #00ff88; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .decision-challenge { background: #1a3a5c; color: #00d4ff; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .decision-review { background: #4a3a0d; color: #ffaa00; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .decision-block { background: #4d0d0d; color: #ff4444; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .reason-card { background: #1a1a2e; border-left: 3px solid; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0 8px 8px 0; }
    .reason-positive { border-color: #ff4444; }
    .reason-negative { border-color: #00ff88; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"
ARTIFACTS = Path("artifacts/sparkov")

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🛡️ SENTINEL</h1>
    <p>Cost-Aware Fraud Intelligence • Every decision priced in ₹, not optimised for a metric</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: System Status ---
with st.sidebar:
    st.markdown("### ⚡ System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        ready = requests.get(f"{API_URL}/ready", timeout=2).json()
        st.success(f"API: Online • {health['model']}")
        
        cols = st.columns(2)
        cols[0].markdown(f"**Model:** {'✅' if ready.get('model_loaded') else '❌'}")
        cols[1].markdown(f"**Calibrator:** {'✅' if ready.get('calibrator_loaded') else '❌'}")
        cols[0].markdown(f"**Velocity:** {'✅' if ready.get('velocity_store') else '⚠️ Degraded'}")
        cols[1].markdown(f"**Ledger:** {'✅' if ready.get('ledger') else '❌'}")
    except:
        st.error("API Offline — Start uvicorn first")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    try:
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
        st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        st.metric("Savings", f"{metrics.get('savings_vs_approve_all_pct', 0):.1f}%")
        st.metric("Precision", f"{metrics.get('precision_at_op', 0):.1%}")
        st.metric("Recall", f"{metrics.get('recall_at_op', 0):.1%}")
    except:
        pass
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repo](https://github.com/itsTIMUS/sentinel-fraud-detection)")
    st.markdown("Built for Razorpay Buildathon 2026 • Track 02")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Score", "📋 Review Queue", "📊 Cost Analysis", "🔬 Model Deep Dive", "📈 Thompson Sampling"
])

# ============================================================
# TAB 1: Live Score
# ============================================================
with tab1:
    st.markdown("### Score a Transaction")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("##### Transaction Details")
        r1c1, r1c2 = st.columns(2)
        amt = r1c1.number_input("Amount (₹)", min_value=0.01, value=500.0, step=10.0)
        category = r1c2.selectbox("Category", [
            "shopping_net", "misc_net", "grocery_pos", "shopping_pos",
            "gas_transport", "misc_pos", "grocery_net", "travel",
            "entertainment", "personal_care", "kids_pets", "food_dining",
            "home", "health_fitness"
        ])
        
        r2c1, r2c2 = st.columns(2)
        trans_time = r2c1.text_input("Time", "2020-06-21 12:14:25")
        cc_num = r2c2.number_input("Card #", value=2703186189652095, step=1)
        
        r3c1, r3c2 = st.columns(2)
        lat = r3c1.number_input("Customer Lat", value=33.9659, format="%.4f")
        long_val = r3c2.number_input("Customer Long", value=-80.9355, format="%.4f")
        
        r4c1, r4c2 = st.columns(2)
        merch_lat = r4c1.number_input("Merchant Lat", value=33.986391, format="%.6f")
        merch_long = r4c2.number_input("Merchant Long", value=-81.200714, format="%.6f")
        
        unix_time = st.number_input("Unix Time", value=1371816865, step=1)
        merchant = st.text_input("Merchant", "fraud_Kirlin and Sons")
        
        score_btn = st.button("⚡ Score Transaction", type="primary", use_container_width=True)
    
    with col_result:
        if score_btn:
            payload = {
                "trans_date_trans_time": trans_time,
                "cc_num": int(cc_num), "merchant": merchant,
                "category": category, "amt": float(amt),
                "first": "Test", "last": "User", "gender": "M",
                "street": "123 Test St", "city": "Columbia", "state": "SC",
                "zip": 29209, "lat": lat, "long": long_val,
                "city_pop": 333497, "job": "Engineer", "dob": "1985-01-15",
                "trans_num": f"test_{np.random.randint(100000)}",
                "unix_time": int(unix_time),
                "merch_lat": merch_lat, "merch_long": merch_long,
            }
            try:
                resp = requests.post(f"{API_URL}/v1/score", json=payload, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Decision badge
                    badge_class = f"decision-{data['decision'].lower()}"
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="{badge_class}">{data['decision']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f"""<div class="metric-card">
                        <div class="value">{data['risk_probability']:.4f}</div>
                        <div class="label">Fraud Probability</div>
                    </div>""", unsafe_allow_html=True)
                    m2.markdown(f"""<div class="metric-card">
                        <div class="value">{data['latency_ms']:.1f}ms</div>
                        <div class="label">Latency</div>
                    </div>""", unsafe_allow_html=True)
                    m3.markdown(f"""<div class="metric-card">
                        <div class="value">₹{data.get('expected_profit_inr', 0):,.0f}</div>
                        <div class="label">Expected Profit</div>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Cost comparison chart
                    costs_data = {
                        "Action": ["ALLOW", "CHALLENGE", "REVIEW", "BLOCK"],
                        "Cost": [
                            data["expected_loss_if_allowed_inr"],
                            data.get("expected_loss_if_challenged_inr", 0),
                            data["expected_loss_if_reviewed_inr"],
                            data["expected_loss_if_blocked_inr"],
                        ],
                    }
                    colors = ["#00ff88", "#00d4ff", "#ffaa00", "#ff4444"]
                    chosen = data["decision"]
                    bar_colors = [c if a == chosen else "rgba(100,100,100,0.3)" for a, c in zip(costs_data["Action"], colors)]
                    
                    fig = go.Figure(data=[go.Bar(
                        x=costs_data["Action"], y=costs_data["Cost"],
                        marker_color=bar_colors,
                        text=[f"₹{c:,.0f}" for c in costs_data["Cost"]],
                        textposition="outside",
                    )])
                    fig.update_layout(
                        title="Expected ₹ Cost Per Action",
                        height=300, margin=dict(t=40, b=20, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8892b0",
                        yaxis_title="₹ Cost",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Reason codes
                    st.markdown("##### Why This Decision")
                    for rc in data.get("reason_codes", []):
                        direction = "positive" if rc["contribution"] > 0 else "negative"
                        icon = "🔴" if rc["contribution"] > 0 else "🟢"
                        st.markdown(f"""
                        <div class="reason-card reason-{direction}">
                            {icon} <strong>{rc['code']}</strong> ({rc['contribution']:+.4f})<br>
                            <span style="color: #8892b0;">{rc['detail']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Degraded warning
                    if data.get("degraded"):
                        st.warning("⚠️ Scored in DEGRADED mode — velocity store unavailable")
                    
                    with st.expander("Raw JSON"):
                        st.json(data)
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Start uvicorn first.")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #8892b0;">
                <p style="font-size: 3rem;">🎯</p>
                <p>Click <strong>Score Transaction</strong> to see the decision</p>
                <p style="font-size: 0.85rem;">The system computes ₹ cost for each action and picks the cheapest</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: Review Queue
# ============================================================
with tab2:
    st.markdown("### Review Queue")
    st.markdown("Transactions flagged for manual analyst review — sorted by risk")
    
    try:
        conn = sqlite3.connect("data/audit.db")
        reviews = pd.read_sql_query(
            """SELECT decision_id, timestamp, amount, risk_probability, decision, 
                      model_version, latency_ms, degraded, holdout_allowed
               FROM decisions 
               ORDER BY timestamp DESC LIMIT 100""",
            conn,
        )
        conn.close()
        
        if len(reviews) > 0:
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Decisions", f"{len(reviews):,}")
            c2.metric("ALLOW", f"{(reviews['decision'] == 'ALLOW').sum():,}")
            c3.metric("CHALLENGE", f"{(reviews['decision'] == 'CHALLENGE').sum():,}")
            review_block = (reviews['decision'] == 'REVIEW').sum() + (reviews['decision'] == 'BLOCK').sum()
            c4.metric("REVIEW + BLOCK", f"{review_block:,}")
            
            st.markdown("---")
            
            # Filter
            filter_col = st.selectbox("Filter by decision:", ["All", "ALLOW", "CHALLENGE", "REVIEW", "BLOCK"])
            if filter_col != "All":
                reviews = reviews[reviews["decision"] == filter_col]
            
            st.dataframe(
                reviews.style.format({
                    "amount": "₹{:.2f}",
                    "risk_probability": "{:.4f}",
                    "latency_ms": "{:.1f}",
                }).applymap(
                    lambda v: "color: #00ff88" if v == "ALLOW" 
                    else "color: #00d4ff" if v == "CHALLENGE"
                    else "color: #ffaa00" if v == "REVIEW" 
                    else "color: #ff4444" if v == "BLOCK" else "",
                    subset=["decision"]
                ),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No decisions recorded yet. Score some transactions first.")
    except Exception as e:
        st.error(f"Could not load ledger: {e}")

# ============================================================
# TAB 3: Cost Analysis
# ============================================================
with tab3:
    st.markdown("### Cost Analysis")
    
    col_slider, col_chart = st.columns([1, 2])
    
    with col_slider:
        st.markdown("##### Adjust Parameters")
        cb_fee = st.slider("Chargeback Fee (₹)", 500, 3000, 1500, 100)
        margin = st.slider("Gross Margin", 0.05, 0.40, 0.18, 0.01)
        friction = st.slider("Friction Cost (₹)", 50, 500, 250, 25)
        ch_success = st.slider("Challenge Success", 0.50, 0.99, 0.85, 0.01)
        ch_dropout = st.slider("Fraudster 3DS Dropout", 0.50, 0.99, 0.95, 0.01)
        
        st.markdown("---")
        st.markdown("##### Comparison")
        st.markdown("""
        | Strategy | ₹ Cost | Savings |
        |---|---|---|
        | Approve All | ₹4,350,825 | — |
        | Fixed t=0.21 | ₹450,242 | 89.7% |
        | **Sentinel** | **₹229,434** | **94.7%** |
        | Oracle | ₹0 | 100% |
        """)
    
    with col_chart:
        # Break-even curve with slider values
        amounts_range = np.linspace(10, 30000, 300)
        thresholds = []
        for a in amounts_range:
            A = a + cb_fee
            B = margin * a + friction + 0.04 * 6000
            thresholds.append(B / (A + B))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=amounts_range, y=thresholds, mode="lines",
            name="Break-even", line=dict(color="#00d4ff", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=amounts_range, y=[1.0] * len(amounts_range), fill="tonexty",
            name="BLOCK zone", fillcolor="rgba(255, 68, 68, 0.1)", line=dict(width=0),
        ))
        fig.add_trace(go.Scatter(
            x=amounts_range, y=[0.0] * len(amounts_range), fill="tonexty",
            name="ALLOW zone", fillcolor="rgba(0, 255, 136, 0.1)", line=dict(width=0),
        ))
        
        # Annotate key points
        for amt_pt in [500, 5000, 20000]:
            A = amt_pt + cb_fee
            B = margin * amt_pt + friction + 0.04 * 6000
            p = B / (A + B)
            fig.add_annotation(x=amt_pt, y=p, text=f"₹{amt_pt:,}: p={p:.3f}",
                               showarrow=True, arrowhead=2, font=dict(size=11))
        
        fig.update_layout(
            title="Break-Even Curve: Where ALLOW Meets BLOCK<br><sub>Small → relaxed | Large → paranoid | Move sliders to shift</sub>",
            xaxis_title="Transaction Amount (₹)", yaxis_title="Fraud Probability Threshold",
            height=500, yaxis_range=[0, 0.5],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8892b0",
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: Model Deep Dive
# ============================================================
with tab4:
    st.markdown("### Model Deep Dive")
    
    try:
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
        
        # Performance metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card">
            <div class="value">{metrics.get('pr_auc', 0):.4f}</div>
            <div class="label">PR-AUC</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="value">{metrics.get('precision_at_op', 0):.1%}</div>
            <div class="label">Precision</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="value">{metrics.get('recall_at_op', 0):.1%}</div>
            <div class="label">Recall</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card">
            <div class="value">{metrics.get('brier_score_calibrated', 0):.6f}</div>
            <div class="label">Brier Score</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("##### Model Progression")
            progression = pd.DataFrame({
                "Model": ["Approve All", "Rules", "LR", "LightGBM (raw)", "LightGBM (calibrated)", "Sentinel v2 (4-action)"],
                "₹ Cost": [4350825, 5290381, 18289845, 338557, 309487, 229434],
                "PR-AUC": [0, 0.35, 0.32, 0.97, 0.95, 0.95],
            })
            
            fig = go.Figure(data=[go.Bar(
                x=progression["Model"], y=progression["₹ Cost"],
                marker_color=["#ff4444", "#ff6b6b", "#ff8888", "#ffaa00", "#00d4ff", "#00ff88"],
                text=[f"₹{c:,.0f}" for c in progression["₹ Cost"]],
                textposition="outside",
            )])
            fig.update_layout(
                height=350, margin=dict(t=20, b=80),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8892b0", yaxis_title="₹ Cost",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_r:
            st.markdown("##### Dual-Dataset Validation")
            comparison = pd.DataFrame({
                "Metric": ["PR-AUC", "Precision", "Recall", "Savings"],
                "Sparkov": ["0.9513", "94.8%", "95.6%", "94.7%"],
                "IEEE-CIS": ["0.4969", "86.9%", "57.5%", "75.2%"],
            })
            st.table(comparison)
            
            st.markdown("""
            **Why IEEE-CIS is lower:** Real anonymized data with adversarial fraudsters. 
            Lower PR-AUC is expected. The cost-aware policy still saves 75.2% — 
            proving the architecture works regardless of model accuracy.
            """)
        
        st.markdown("---")
        
        st.markdown("##### Architecture")
        st.markdown("""
        **Intended use:** Real-time fraud scoring for card-not-present payment transactions  
        **Not for:** Credit scoring, identity verification, law enforcement  
        
        **Limitations:**
        - Trained on simulated data (Sparkov) — real fraud patterns may differ
        - Velocity features need history — new cards score with defaults
        - Cost parameters are industry estimates — should be tuned per merchant
        - No concept drift detection — model should be periodically retrained
        """)
    
    except Exception as e:
        st.error(f"Could not load metrics: {e}")

# ============================================================
# TAB 5: Thompson Sampling
# ============================================================
with tab5:
    st.markdown("### Thompson Sampling — Self-Learning Cost Parameters")
    st.markdown("""
    In production, Sentinel wouldn't use fixed cost estimates forever. 
    Thompson Sampling lets the system **learn from observed outcomes** — 
    starting from wrong guesses and converging to true values.
    """)
    
    # Show convergence charts if they exist
    convergence_path = Path("reports/plots/thompson_convergence.png")
    cumulative_path = Path("reports/plots/thompson_cumulative_cost.png")
    
    if convergence_path.exists():
        st.image(str(convergence_path), caption="Parameters converge from wrong priors to true values")
    
    if cumulative_path.exists():
        st.image(str(cumulative_path), caption="Exploration cost shrinks as parameters converge")
    
    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### ✅ Learnable Parameters")
        st.markdown("""
        | Parameter | Feedback Signal |
        |---|---|
        | Chargeback fee | Bank statement |
        | Challenge success rate | OTP completion |
        | Fraudster 3DS dropout | Challenge outcome |
        | Analyst catch rate | Analyst decisions |
        """)
    
    with col_r:
        st.markdown("##### ❌ Not Learnable (Need Experiments)")
        st.markdown("""
        | Parameter | Why Not |
        |---|---|
        | Churn probability | Takes months to observe |
        | Customer LTV | Takes months/years |
        | Friction cost | Not directly measurable |
        """)
    
    st.info("""
    **The key insight:** Knowing what you CAN'T learn automatically is as important as what you can. 
    The unlearnable parameters need deliberate A/B tests, not passive observation.
    """)