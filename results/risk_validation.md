# Physics-Grounded Hydraulic Risk Layer & TPPI Report

## 1. WNTR Hydraulic Simulation on CTOWN.INP
Simulated component outages across 11 pumps, 4 valves, 7 tanks, and monitored junction manifolds.
Measured exact physical consequences: **Pressure Deficit** (m$\cdot$h) and **Affected Consumers** across the network.

### Top-10 Critical Cyber-Physical Assets (Validated Base Risk)

| Rank | Asset ID | Component Type | Structural Criticality | Hydraulic Impact (ISS) | Cascading Reach (CRI) | Susceptibility (ASS) | Base Risk (TDCRI Base) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PU2** | Pump | 0.9069 | 1.0000 | 0.2000 | 0.87 | **0.6883** |
| 2 | **T2** | Tank | 0.7453 | 0.7503 | 0.0000 | 0.75 | **0.5989** |
| 3 | **v1** | Valve | 0.8611 | 0.7762 | 0.0552 | 0.84 | **0.5975** |
| 4 | **T4** | Tank | 0.7410 | 0.8369 | 0.0000 | 0.75 | **0.5847** |
| 5 | **T1** | Tank | 0.7442 | 0.5477 | 0.0000 | 0.75 | **0.5833** |
| 6 | **V47** | Valve | 0.8522 | 0.4819 | 0.0400 | 0.84 | **0.5231** |
| 7 | **V45** | Valve | 0.8520 | 0.3884 | 0.0400 | 0.84 | **0.5107** |
| 8 | **PU6** | Pump | 1.0000 | 0.0500 | 0.0000 | 0.87 | **0.5075** |
| 9 | **PU9** | Pump | 0.9681 | 0.0500 | 0.0000 | 0.87 | **0.5055** |
| 10 | **PU8** | Pump | 0.9616 | 0.0500 | 0.0000 | 0.87 | **0.5051** |

---

## 2. Non-Circular Risk Ranking Validation & Sensitivity Analysis
- **Negative Finding: Pure Structural Centrality Inadequacy**:
  $$\rho(\text{Structural Centrality}, \text{Hydraulic Impact ISS}) = +0.0932 \quad (p = 0.600)$$
  Demonstrates that pure graph topological centrality (betweenness/closeness) alone fails to predict physical consequences, because transmission corridors have low local demand despite immense downstream hydraulic impact.
- **Non-Circular Multi-Dimensional Criticality (MDCI)**:
  $$\rho(\text{Non-Circular MDCI}, \text{Hydraulic Impact ISS}) = +0.1300 \quad (p = 0.464)$$
  Evaluated purely from intrinsic engineering design ratings (tank storage volume, pump power, valve diameter, junction base demand) with zero simulation inputs.
- **Non-Circular Risk Validation Against Independent Spatial Customer Impact**:
  $$\rho(\text{Cyber-Physical Base Risk}, \text{Customer Impact Footprint (Affected Nodes)}) = \mathbf{+0.5805} \quad (p = 3.20e-04)$$
  Statistically validates that our multi-dimensional threat-driven risk formulation strongly predicts the spatial extent of customer service disruption without circularity (`Affected Nodes` is not an input feature to Base Risk).
- **Composite Risk Grounding (Pressure Deficit Alignment)**:
  $$\rho(\text{Cyber-Physical Base Risk}, \text{Hydraulic Impact ISS}) = \mathbf{+0.6109} \quad (p = 1.24e-04)$$

- **Empirical Sensitivity Distribution over 1,000 Draws (Kendall's $\tau$)**:
  - **Distribution**: Mean $\tau = \mathbf{0.9859}$ (Std $= 0.0133$, 5th% $= 0.9572$, 95th% $= 1.0000$, Min $= 0.9501$, Max $= 1.0000$).
  - **Top-5 Asset Retention Frequency**:
    - **PU2**: 100.0%
    - **v1**: 100.0%
    - **T1**: 100.0%
    - **T2**: 100.0%
    - **T4**: 100.0%

---

## 3. Threat-Prioritized Protection Investment (TPPI 0/1 Knapsack)

### A. Tied Groups & Deterministic Tie-Breaking
When components exhibit identical priority densities (e.g. monitored sensor junctions with identical \$40k unit cost and nominal base demand), ties are resolved deterministically by:
1. Priority Density (Protection Value / Cost)
2. Absolute Protection Value
3. Asset ID (lexicographical order)

**Identified Tied Groups**:
- **Junction Group** (6 assets: J422, J289, J302, J307, J317, J306): Unit Cost = \$40k, Mean Priority Density = 0.004946
- **Tank Group** (2 assets: T4, T1): Unit Cost = \$100k, Mean Priority Density = 0.004730
- **Junction Group** (3 assets: J14, J415, J256): Unit Cost = \$40k, Mean Priority Density = 0.004565
- **Junction Group** (2 assets: J269, J280): Unit Cost = \$40k, Mean Priority Density = 0.004417
- **Tank Group** (2 assets: T6, T5): Unit Cost = \$100k, Mean Priority Density = 0.003669
- **Tank Group** (2 assets: T3, T7): Unit Cost = \$100k, Mean Priority Density = 0.003552
- **Pump Group** (5 assets: PU6, PU9, PU8, PU10, PU11): Unit Cost = \$120k, Mean Priority Density = 0.003219
- **Pump Group** (4 assets: PU5, PU3, PU7, PU4): Unit Cost = \$120k, Mean Priority Density = 0.003074

### B. Multi-Budget Allocation Sensitivity (\$200k, \$500k, \$1,000k)

| Budget Tier | Total Spent | Assets Protected | Funded Components | Total Value |
| :---: | :---: | :---: | :---: | :---: |
| **\$200k** | \$200k | 4 | v1, J422, J289, J302 | 1.0243 |
| **\$500k** | \$500k | 9 | v1, J422, J289, J302, J307, J317, J306, T2, V47 | 2.4790 |
| **\$1000k** | \$1000k | 17 | v1, J422, J289, J302, J307, J317, J306, T2, J300, T4, V47, V45, J14, J415, J256, J269, PU2 | 4.7610 |

### C. Detailed \$500k Baseline Protection Allocation

| Selected Asset | Type | Unit Cost | Protection Benefit | TPPI Priority Score |
| :---: | :---: | :---: | :---: | :---: |
| **v1** | Valve | \$80k | 0.80 | 0.005377 |
| **J422** | Junction | \$40k | 0.70 | 0.004955 |
| **J289** | Junction | \$40k | 0.70 | 0.004951 |
| **J302** | Junction | \$40k | 0.70 | 0.004947 |
| **J307** | Junction | \$40k | 0.70 | 0.004942 |
| **J317** | Junction | \$40k | 0.70 | 0.004941 |
| **J306** | Junction | \$40k | 0.70 | 0.004940 |
| **T2** | Tank | \$100k | 0.90 | 0.004851 |
| **V47** | Valve | \$80k | 0.80 | 0.004708 |

- **Total Budget Allocated**: **$500k / $500k**
- **Selected Assets Protected**: **9 components**
