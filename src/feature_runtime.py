import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import __main__


SENTINEL_NUM = -999.0


def _safe_col(df: pd.DataFrame, col: str, default=SENTINEL_NUM) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _safe_str_col(df: pd.DataFrame, col: str, default="MISSING") -> pd.Series:
    if col in df.columns:
        return df[col].astype("string").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="string")


@dataclass
class FeatureMeta:
    amount_95: float
    global_amt_mean: float
    global_amt_std: float
    global_amt_median: float
    card1_amt_mean: dict[str, float]
    card1_amt_std: dict[str, float]
    card1_amt_median: dict[str, float]
    card1_txn_count: dict[str, float]
    card1_addr1_unique: dict[str, float]
    addr1_txn_count: dict[str, float]
    card1_addr1_count: dict[str, float]
    card1_dist1_mean: dict[str, float]
    p_email_freq: dict[str, float]
    device_freq: dict[str, float]
    c1_freq: dict[str, float]
    v_cols: list[str]
    online_stateful_features: list[str]


class FraudFeatureBuilder:
    """
    Train-fitted feature builder for online inference.
    Only online-safe features are truly computed from one request.
    Stateful features can be injected via `context` or default to sentinel.
    """

    def __init__(self, pca_components: int = 2):
        self.pca_components = pca_components
        self.meta: FeatureMeta | None = None
        self.pca: PCA | None = None
        self.feature_order_: list[str] | None = None

    @staticmethod
    def _keyify(x: pd.Series) -> pd.Series:
        return x.astype("string").fillna("MISSING")

    def fit(self, train_df: pd.DataFrame) -> "FraudFeatureBuilder":
        df = train_df.copy()

        amt = _safe_col(df, "TransactionAmt", default=0.0).astype(float)
        card1 = self._keyify(_safe_col(df, "card1"))
        addr1 = self._keyify(_safe_col(df, "addr1"))
        p_email = self._keyify(_safe_str_col(df, "P_emaildomain"))
        device = self._keyify(_safe_str_col(df, "DeviceInfo"))
        c1 = self._keyify(_safe_col(df, "C1"))

        card1_amt_mean = amt.groupby(card1).mean().to_dict()
        card1_amt_std = amt.groupby(card1).std().fillna(0.0).to_dict()
        card1_amt_median = amt.groupby(card1).median().to_dict()
        card1_txn_count = card1.value_counts(dropna=False).astype(float).to_dict()
        card1_addr1_unique = df.assign(card1_key=card1, addr1_key=addr1).groupby("card1_key")["addr1_key"].nunique().astype(float).to_dict()
        addr1_txn_count = addr1.value_counts(dropna=False).astype(float).to_dict()

        pair_key = (card1 + "__" + addr1)
        card1_addr1_count = pair_key.value_counts(dropna=False).astype(float).to_dict()

        dist1 = _safe_col(df, "dist1")
        card1_dist1_mean = dist1.groupby(card1).mean().fillna(SENTINEL_NUM).to_dict()

        p_email_freq = p_email.value_counts(dropna=False).astype(float).to_dict()
        device_freq = device.value_counts(dropna=False).astype(float).to_dict()
        c1_freq = c1.value_counts(dropna=False).astype(float).to_dict()

        v_cols = [c for c in df.columns if c.startswith("V")]
        self.pca = None
        if v_cols:
            self.pca = PCA(n_components=min(self.pca_components, len(v_cols)))
            self.pca.fit(df[v_cols].fillna(SENTINEL_NUM))

        self.meta = FeatureMeta(
            amount_95=float(amt.quantile(0.95)),
            global_amt_mean=float(amt.mean()),
            global_amt_std=float(amt.std() if amt.std() == amt.std() else 0.0),
            global_amt_median=float(amt.median()),
            card1_amt_mean={str(k): float(v) for k, v in card1_amt_mean.items()},
            card1_amt_std={str(k): float(v) for k, v in card1_amt_std.items()},
            card1_amt_median={str(k): float(v) for k, v in card1_amt_median.items()},
            card1_txn_count={str(k): float(v) for k, v in card1_txn_count.items()},
            card1_addr1_unique={str(k): float(v) for k, v in card1_addr1_unique.items()},
            addr1_txn_count={str(k): float(v) for k, v in addr1_txn_count.items()},
            card1_addr1_count={str(k): float(v) for k, v in card1_addr1_count.items()},
            card1_dist1_mean={str(k): float(v) for k, v in card1_dist1_mean.items()},
            p_email_freq={str(k): float(v) for k, v in p_email_freq.items()},
            device_freq={str(k): float(v) for k, v in device_freq.items()},
            c1_freq={str(k): float(v) for k, v in c1_freq.items()},
            v_cols=v_cols,
            online_stateful_features=[
                "TimeSinceLastTransaction",
                "TransactionVelocity1h",
                "TransactionVelocity24h",
            ],
        )
        return self

    def _map_or_default(self, key_series: pd.Series, mapping: dict[str, float], default: float) -> pd.Series:
        return key_series.astype("string").map(mapping).fillna(default)

    def transform(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if self.meta is None:
            raise ValueError("Feature builder is not fitted.")

        context = context or {}
        out = df.copy()

        amt = _safe_col(out, "TransactionAmt", default=0.0).astype(float)
        card1 = self._keyify(_safe_col(out, "card1"))
        addr1 = self._keyify(_safe_col(out, "addr1"))
        p_email = self._keyify(_safe_str_col(out, "P_emaildomain"))
        device = self._keyify(_safe_str_col(out, "DeviceInfo"))
        c1 = self._keyify(_safe_col(out, "C1"))
        transaction_dt = _safe_col(out, "TransactionDT", default=0.0).astype(float)
        d1 = _safe_col(out, "D1")
        d2 = _safe_col(out, "D2")
        d15 = _safe_col(out, "D15")
        dist1 = _safe_col(out, "dist1")
        addr2 = _safe_col(out, "addr2")
        p_emaildomain = _safe_str_col(out, "P_emaildomain")
        r_emaildomain = _safe_str_col(out, "R_emaildomain")
        device_type = _safe_str_col(out, "DeviceType")

        card1_amt_mean = self._map_or_default(card1, self.meta.card1_amt_mean, self.meta.global_amt_mean)
        card1_amt_std = self._map_or_default(card1, self.meta.card1_amt_std, self.meta.global_amt_std)
        card1_amt_median = self._map_or_default(card1, self.meta.card1_amt_median, self.meta.global_amt_median)

        out["TransactionAmt_Log"] = np.log1p(np.clip(amt, a_min=0, a_max=None))
        out["Amt_decimal"] = np.mod(amt, 1.0)
        out["card1_Amt_mean"] = card1_amt_mean
        out["card1_Amt_std"] = card1_amt_std
        out["card1_Amt_median"] = card1_amt_median
        out["AmountDeviationUser"] = amt / (card1_amt_mean + 1e-3)
        out["AmountStdScore"] = (amt - card1_amt_mean) / (card1_amt_std + 1e-3)
        out["AmountToMedianRatio"] = amt / (card1_amt_median + 1e-3)
        out["IsLargeTransaction"] = (amt > self.meta.amount_95).astype(int)
        out["IsSmallTestTransaction"] = (amt < 5).astype(int)

        out["TransactionHour"] = (transaction_dt / 3600) % 24
        out["TransactionDayOfWeek"] = (transaction_dt / 86400) % 7
        out["IsNightTransaction"] = out["TransactionHour"].between(0, 5).astype(int)
        out["IsWeekendTransaction"] = (out["TransactionDayOfWeek"] >= 5).astype(int)

        # Stateful features: use external context if available, else sentinel
        out["TimeSinceLastTransaction"] = float(context.get("TimeSinceLastTransaction", SENTINEL_NUM))
        out["TransactionVelocity1h"] = float(context.get("TransactionVelocity1h", SENTINEL_NUM))
        out["TransactionVelocity24h"] = float(context.get("TransactionVelocity24h", SENTINEL_NUM))

        out["CardTransactionCount"] = self._map_or_default(card1, self.meta.card1_txn_count, 1.0)
        out["CardissuerFrequency"] = out["CardTransactionCount"]
        out["DaysSinceRegistration"] = d1.fillna(SENTINEL_NUM)
        out["AccountAgeRisk"] = 1.0 / (out["DaysSinceRegistration"] + 1.0)
        out["TimeSinceLastPurchase"] = d2.fillna(SENTINEL_NUM)
        out["RegistrationToTransactionGap"] = transaction_dt - out["DaysSinceRegistration"]
        d15_mean_by_card = self._map_or_default(card1, self.meta.card1_amt_mean, self.meta.global_amt_mean)
        out["D15_to_Mean_card1"] = d15.fillna(SENTINEL_NUM) / (d15_mean_by_card + 1e-3)

        out["AddrMismatch"] = (addr1 != self._keyify(addr2)).astype(int)
        out["AddressTransactionCount"] = self._map_or_default(addr1, self.meta.addr1_txn_count, 1.0)
        out["CardAddressCombination"] = self._map_or_default(card1 + "__" + addr1, self.meta.card1_addr1_count, 1.0)
        dist1_mean_by_card = self._map_or_default(card1, self.meta.card1_dist1_mean, SENTINEL_NUM)
        out["DistanceDeviation"] = dist1.fillna(SENTINEL_NUM) - dist1_mean_by_card
        out["IsLongDistance"] = (dist1.fillna(SENTINEL_NUM) > 100).astype(int)
        dist1_filled = dist1.fillna(SENTINEL_NUM)
        dist1_min, dist1_max = float(dist1_filled.min()), float(dist1_filled.max())
        denom = (dist1_max - dist1_min) if dist1_max != dist1_min else 1.0
        out["DistanceRiskScore"] = (dist1_filled - dist1_min) / denom
        out["card1_addr1_unique"] = self._map_or_default(card1, self.meta.card1_addr1_unique, 1.0)

        out["EmailDomainMatch"] = (p_emaildomain == r_emaildomain).astype(int)
        out["IsAnonymousEmail"] = p_emaildomain.isin(["protonmail.com", "mail.com"]).astype(int)
        out["EmailDomainFrequency"] = self._map_or_default(p_email, self.meta.p_email_freq, 1.0)

        out["CardIPCount"] = _safe_col(out, "C5").fillna(SENTINEL_NUM)
        out["AddressDeviceCount"] = _safe_col(out, "C7").fillna(SENTINEL_NUM)
        out["AssociationRatio"] = (_safe_col(out, "C1").fillna(0.0) + _safe_col(out, "C2").fillna(0.0)) / (
            _safe_col(out, "C3").fillna(0.0) + 0.01
        )
        out["TotalAssociations"] = (
            _safe_col(out, "C1").fillna(0.0)
            + _safe_col(out, "C2").fillna(0.0)
            + _safe_col(out, "C3").fillna(0.0)
        )
        out["IsMobileDevice"] = (device_type == "mobile").astype(int)
        out["null_counts"] = out.isnull().sum(axis=1)
        if "id_31" in out.columns:
            out["id_31_device"] = out["id_31"].astype("string").str.split(" ").str[0].fillna("MISSING")
        out["Device_Freq"] = self._map_or_default(device, self.meta.device_freq, 1.0)
        out["C1_count"] = self._map_or_default(c1, self.meta.c1_freq, 1.0)

        if self.pca is not None and self.meta.v_cols:
            v_block = pd.DataFrame(index=out.index)
            for c in self.meta.v_cols:
                v_block[c] = _safe_col(out, c).fillna(SENTINEL_NUM)
            v_pca = self.pca.transform(v_block[self.meta.v_cols])
            out["V_PCA_1"] = v_pca[:, 0]
            if v_pca.shape[1] > 1:
                out["V_PCA_2"] = v_pca[:, 1]

        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def save(self, path: str | Path) -> None:
        payload = {
            "pca_components": self.pca_components,
            "meta": self.meta,
            "pca": self.pca,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "FraudFeatureBuilder":
        # Register old module path for backward compatibility with pickled artifacts
        current_module = sys.modules[__name__]
        sys.modules['feature_runtime'] = current_module
        __main__.FraudFeatureBuilder = cls
        
        try:
            payload = joblib.load(path)
        finally:
            # Clean up if needed
            if 'feature_runtime' in sys.modules and sys.modules['feature_runtime'] is current_module:
                pass  # Keep it registered for subsequent loads
        
        obj = cls(pca_components=payload["pca_components"])
        obj.meta = payload["meta"]
        obj.pca = payload["pca"]
        return obj