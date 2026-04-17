# P33 Protocol

P33 uses rolling-origin evaluation as a non-negotiable protocol. Every forecast origin is evaluated using only information available up to that origin. No future observations may enter feature construction, scaling, transformation, model fitting, threshold estimation, or evaluation inputs.

Preprocessing must be train-only whenever statistics are estimated inside a split. This includes any normalization, imputation, threshold calibration, or feature standardization.

The project uses daily frequency and `Hmax = 7` days. Horizon-wise evaluation is mandatory. Persistence is the required baseline. Seasonal persistence may be used only when the temporal structure justifies a seasonal carry-forward benchmark.

Interpretation is based on the joint reading of `skill` and `alpha`:

- positive `skill` with `alpha` near 1 suggests more credible dynamic usefulness
- positive `skill` with low `alpha` suggests attenuated dynamics and plausible ghost skill
- `alpha` clearly above 1 indicates variance inflation and should also be audited

The paper distinguishes robust skill from plausible ghost skill by requiring that forecast improvement be read together with variance retention rather than from error reduction alone.
