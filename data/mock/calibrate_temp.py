import numpy as np

# Current RISK_BANDS boundaries
# We need: f(PD%) -> Score such that:
#   f(2%) = 720, f(8%) = 640, f(18%) = 560, f(35%) = 460
# Using log model: Score = a + b * ln(PD/100)

pds = [2, 8, 18, 35]
scores = [720, 640, 560, 460]
ln_pds = [np.log(p / 100) for p in pds]

# Linear regression: Score = intercept + slope * ln(PD/100)
coeffs = np.polyfit(ln_pds, scores, 1)
slope = coeffs[0]
intercept = coeffs[1]

print(f"Best fit: Score = {intercept:.2f} + {slope:.2f} * ln(PD/100)")
print(f"  slope (b) = {slope:.4f}")
print(f"  intercept (a) = {intercept:.4f}")
print()

print("Verification (new formula):")
for pd in [0.1, 0.5, 1, 2, 5, 8, 12, 18, 25, 35, 49, 50, 65, 80, 100]:
    s = slope * np.log(pd / 100) + intercept
    s_clamped = max(300, min(850, s))
    print(f"  PD={pd:6.1f}% -> Score={s_clamped:.0f}")

print()
print("Comparison OLD vs NEW:")
for pd in [2, 8, 18, 35, 49]:
    old = 850 + 93 * np.log(0.02) - 93 * np.log(pd / 100)
    new = slope * np.log(pd / 100) + intercept
    print(f"  PD={pd}%: OLD={old:.0f}, NEW={new:.0f}")
