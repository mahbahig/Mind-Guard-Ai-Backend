# 🧪 Feature Extraction Testing Checklist

Use this checklist to verify that the mathematical features in [[Feature_Engineering]] are evaluated correctly against known small array outputs.

## Statistical Features
- [ ] **MAD (Mean Absolute Deviation):** 
  - Array: `[1, 2, 3, 4, 5]` -> Expect `1.2`
- [ ] **IQR (Interquartile Range):** 
  - Array: `[1, 2, 3, 4, 5]` -> Expect `2.0`
- [ ] **RMS (Root Mean Square):** 
  - Array: `[1, 2, 3, 4, 5]` -> Expect `~3.3166`

## Nonlinear Features
- [ ] **ACL (Autocorrelation Lag 1):** 
  - Test with a perfect sine wave. Expect a very high positive value near `1.0` (depending on frequency).
  - Test with random noise. Expect near `0.0`.
- [ ] **SF (Shannon Entropy):** 
  - Test with a uniform random distribution vs a constant flat line. The uniform distribution should have maximum entropy, flat line should be close to `0`.

## 🐛 Known Issues/Bugs
*(Log any failures or unexpected outputs during feature verification testing here.)*
