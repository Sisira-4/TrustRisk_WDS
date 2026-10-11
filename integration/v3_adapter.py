"""V3 Residual Adapter: maps per-sensor z-scores to asset-level attack probabilities.

Calibrated strictly on OUT-OF-FOLD cross-validation scores from dataset04.
Does NOT claim localization accuracy: dataset04 does not provide ground-truth per-asset attack labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Mapping file
MAPPING_PATH = Path(__file__).resolve().parents[1] / "network" / "mappings" / "batadal_epanet_mapping.csv"


class V3ResidualAdapter:
    """
    Adapter that ingests exported residual z-score CSVs (from export_scores.py)
    and produces asset-level attack probability signals.
    
    Calibration is performed strictly on OUT-OF-FOLD CV scores (Platt scaling).
    """

    def __init__(self, mapping_path=MAPPING_PATH):
        self.mapping_path = Path(mapping_path)
        self.sensor_to_asset = {}
        self.asset_to_sensors = {}
        self._load_mapping()
        self.calibrator = None

    def _load_mapping(self):
        if self.mapping_path.exists():
            df = pd.read_csv(self.mapping_path)
            for _, row in df.iterrows():
                var = row["variable"].strip()
                elem = row["element"].strip()
                self.sensor_to_asset[var] = elem
                self.asset_to_sensors.setdefault(elem, []).append(var)

    def fit_calibrator_oof(self, oof_scores, oof_labels):
        """Fits a Platt calibrator strictly on out-of-fold validation scores."""
        scores = np.asarray(oof_scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(oof_labels, dtype=int)
        self.calibrator = LogisticRegression(C=1.0)
        self.calibrator.fit(scores, labels)
        return self

    def predict_asset_probabilities(self, scores_df):
        """
        Takes scores DataFrame (with 'score' and 'z_<sensor>' columns) and returns
        a DataFrame of calibrated asset attack probabilities per timestep.

        Disclaimer: dataset04 provides only system-level ATT_FLAG labels, not
        asset-level targets. Sensor-to-asset probabilities reflect sensor anomaly
        concentrations and do NOT claim verified physical localization accuracy.
        """
        if self.calibrator is None:
            # Fallback sigmoid if not fitted
            global_prob = 1.0 / (1.0 + np.exp(-0.1 * scores_df["score"].values))
        else:
            global_prob = self.calibrator.predict_proba(scores_df["score"].values.reshape(-1, 1))[:, 1]

        asset_probs = pd.DataFrame(index=scores_df.index)
        asset_probs["global_p_attack"] = global_prob

        # Map available z-scores to assets
        z_cols = [c for c in scores_df.columns if c.startswith("z_")]
        for asset, sensors in self.asset_to_sensors.items():
            avail = [f"z_{s}" for s in sensors if f"z_{s}" in z_cols]
            if avail:
                # Local asset z-magnitude (max positive z across sensors on this asset)
                local_z = scores_df[avail].values
                max_local_z = np.maximum(0.0, np.max(local_z, axis=1))
                # Modulate global probability with localized z-intensity
                local_weight = 1.0 / (1.0 + np.exp(-np.clip(max_local_z - 3.0, -10, 10)))
                asset_probs[f"asset_{asset}_p"] = np.clip(global_prob * (0.5 + 0.5 * local_weight), 0.0, 1.0)

        return asset_probs

    def run_tdcri_with_integrator(self, integrator, scores_df, node_id="T1"):
        """
        Feeds V3 calibrated probabilities to TrustRiskIntegrator without modifying that class.
        """
        asset_probs = self.predict_asset_probabilities(scores_df)
        p_attack = asset_probs["global_p_attack"].values
        conf = np.maximum(p_attack, 1.0 - p_attack)
        unc = 0.1  # residual uncertainty
        ers = 0.9  # explanation reliability

        tdcri_series = []
        for c in conf:
            ts = integrator.compute_trust_score(conf=float(c), uncertainty=unc, ers=ers)
            td = integrator.compute_tdcri(trust_score=ts, affected_node_id=node_id)
            tdcri_series.append(td)

        return np.array(tdcri_series), asset_probs
