# Open Questions & Unverified Items (`docs/OPEN_QUESTIONS.md`)

This document records research items, assumptions, or future explorations that cannot be automatically verified or that require human domain consultation.

---

## 1. Domain Cost Schedules & Utility Assumptions
- **Unit Hardening Costs**: In `config/trustrisk.yaml`, unit costs for physical hardening are assumed as follows:
  - Pumps: $120k (Secure PLC, physical enclosure, actuator authentication)
  - Valves: $80k (Telemetry encryption, dual-redundant solenoids)
  - Tanks: $100k (Multi-probe level sensing, physical fencing)
  - Junctions: $40k (Cryptographic pressure tap)
- **Question for Utility Stakeholders**: Do these cost estimates reflect current procurement and municipal bidding schedules for water utilities of comparable scale to C-Town (~400 nodes, 100k population)?
- **Sensitivity Context**: Section 5 demonstrates that Top-10 asset ranking is highly robust to uniform and single-weight perturbations ($\tau = 0.9769$), but utility-specific procurement constraints should be reviewed prior to deployment.

## 2. Target Venue & Formatting
- **Question**: Which journal is the primary target for initial submission?
  - *IEEE TIFS*: Focus on cyber-physical threat models, adversarial false data injection, and selective prediction theory.
  - *IEEE IoT-J*: Focus on edge SCADA integration, telemetry calibration, and Streamlit operational decision support.
  - *Water Resources Research (WRR)*: Focus on WNTR hydraulic consequence modeling, pressure deficit empirical grounding, and municipal asset resilience.

## 3. Second Dataset Evaluation (Future Work)
- **Status**: Quarantined single-dataset evaluation on BATADAL (C-Town) is fully complete and verified.
- **Question**: Should the framework be cross-evaluated on the WADI (Water Distribution) testbed or synthetic WNTR attack scenarios as a post-submission extension?
- **Current Position**: Honestly documented under Future Work in [`docs/MANUSCRIPT_REVISION.md`](file:///c:/Users/shino/Desktop/Project/docs/MANUSCRIPT_REVISION.md) to maintain strictly frozen test quarantine on the primary benchmark.

## 4. Operational Latency Acceptance by Municipal Utilities
- **Observation**: Trust validation introduces a mean response delay of 6.57 hours (Policy 3), 5.71 hours (Policy 4), and 6.86 hours (Policy 5) across all attacks vs. 4.86 hours under baseline persistence.
- **Question**: Is an additional 1.7-2.3 hour latency acceptable to water operators for low-magnitude stealthy attacks, in exchange for routing 3 of 3 false-alarm events to analyst review rather than executing automated pump shutoffs?
- **Analysis**: The manuscript explicitly discusses this trade-off: fast-acting burst attacks (e.g., Attack 7) are detected quickly (delay = 1–2 hours), whereas subtle tank drainage manipulations accumulate over multi-day periods and require observation accumulation.
