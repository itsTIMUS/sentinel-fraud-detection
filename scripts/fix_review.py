"""Find the right review_delay_churn_inr so REVIEW isn't dead."""

import sys
sys.path.insert(0, ".")
import yaml
import numpy as np
from src.sentinel.cost import expected_cost_allow, expected_cost_block, expected_cost_review, expected_cost_challenge

# Load base costs
with open("config/costs.yaml") as f:
    base_costs = yaml.safe_load(f)

print("Testing different review_delay_churn_inr values...\n")
print(f"{'delay':>6s} | {'₹500 p=0.15':>20s} | {'₹5000 p=0.10':>20s} | {'₹20000 p=0.08':>20s}")
print("-" * 75)

for delay in [0, 10, 20, 30, 40, 50, 60, 80]:
    costs = base_costs.copy()
    costs["review_delay_churn_inr"] = delay
    
    results = []
    for amt, p in [(500, 0.15), (5000, 0.10), (20000, 0.08)]:
        ca = expected_cost_allow(p, amt, costs)
        cc = expected_cost_challenge(p, amt, costs)
        cr = expected_cost_review(p, amt, costs)
        cb = expected_cost_block(p, amt, costs)
        
        options = {"ALLOW": ca, "CHALLENGE": cc, "REVIEW": cr, "BLOCK": cb}
        winner = min(options, key=options.get)
        results.append(f"{winner:>9s} (₹{options[winner]:>6.0f})")
    
    print(f"{delay:>6d} | {results[0]} | {results[1]} | {results[2]}")

# Now show full decision distribution for a few delay values
print("\n\nFull decision table at delay=30:")
costs = base_costs.copy()
costs["review_delay_churn_inr"] = 30

print(f"\n{'p':>6s} {'amt':>7s} | {'ALLOW':>8s} {'CHALLENGE':>10s} {'REVIEW':>8s} {'BLOCK':>8s} | {'Winner':>10s}")
print("-" * 75)
for p in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    for amt in [500, 2000, 5000, 10000, 20000]:
        ca = expected_cost_allow(p, amt, costs)
        cc = expected_cost_challenge(p, amt, costs)
        cr = expected_cost_review(p, amt, costs)
        cb = expected_cost_block(p, amt, costs)
        
        options = {"ALLOW": ca, "CHALLENGE": cc, "REVIEW": cr, "BLOCK": cb}
        winner = min(options, key=options.get)
        print(f"{p:>6.2f} {amt:>7d} | {ca:>8.0f} {cc:>10.0f} {cr:>8.0f} {cb:>8.0f} | {winner:>10s}")
    print()