# 🛡️ Sentinel — Cost-Aware Fraud Detection for Payment Gateways

> Most fraud systems optimise a metric. Sentinel optimises rupees — it computes, per transaction, whether blocking costs more than the fraud would, and it can show you the audit trail for every decision it has ever made.

## The Problem

A merchant runs an online store. Some payments are stolen cards. When fraud goes through, the cardholder complains, the bank claws money back, and the merchant loses the goods, the money, and pays a chargeback penalty fee.

The obvious fix — block anything suspicious — is worse than the disease. Block a real customer at checkout and you lose the sale, possibly the customer forever.

**Sentinel is the dial between "lose money to thieves" and "lose money by insulting customers" — set automatically, priced in ₹.**

## How It Works

For every payment, in under 50 milliseconds, Sentinel:

1. Examines context — not just "₹40,000 is big" but "₹40,000 from a card that never spent over ₹3,000, at 3am, 700km from usual location, 4th attempt in 10 minutes"
2. Estimates a calibrated probability of fraud
3. Computes expected ₹ loss if allowed vs blocked — approves, challenges, or blocks whichever is cheapest
4. Records why — permanently, in language a support agent can read

## Defense-Only Compliance

This system is strictly defense-only. No fraud-generation tooling, no evasion testing, no adversarial attack simulation. Robustness is evaluated through distribution shift and concept drift analysis only.

## Status

🚧 Under construction — Razorpay Buildathon Track 02 submission.