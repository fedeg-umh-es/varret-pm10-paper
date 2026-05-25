# Model-Family Diagnostic Summary Table

| Model | Family | Med Skill | Med Alpha | Collapse % | Retained % | Near-Ideal % | Sig Imp | Sig Deg | Med CSI | Med FAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgb_direct | Direct ML | 0.205 | 0.151 | 99.2% | 0.0% | 0.0% | 111 | 0 | 0.121 | 0.647 |
| ridge_direct | Direct ML | 0.219 | 0.087 | 99.2% | 0.0% | 0.0% | 117 | 0 | 0.104 | 0.595 |
| sarima | Statistical baseline | 0.208 | 0.095 | 92.4% | 0.0% | 0.0% | 44 | 0 | 0.118 | 0.500 |
| seasonal_naive | Variance-preserving naive | -0.026 | 1.000 | 0.0% | 98.3% | 16.8% | 5 | 34 | 0.111 | 0.822 |
| stl_ridge_direct | Decomposition + Ridge | -1.107 | 1.399 | 0.0% | 100.0% | 0.0% | 0 | 119 | 0.104 | 0.896 |
