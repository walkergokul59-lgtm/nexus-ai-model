from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import math
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from collections import Counter


# ----------------------------
# Log type detection
# ----------------------------
LOG_KINDS = ("auth", "netflow", "dns", "http")

def detect_log_kind(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)

    if {"user", "app", "auth_method", "outcome", "src_ip", "timestamp"}.issubset(cols):
        return "auth"
    if {"src_ip", "dst_ip", "proto", "dst_port", "bytes", "packets", "timestamp"}.issubset(cols):
        return "netflow"
    if {"src_ip", "query", "qtype", "rcode", "timestamp"}.issubset(cols):
        return "dns"
    if {"src_ip", "host", "method", "path", "status", "timestamp"}.issubset(cols):
        return "http"
    
    # Fallback or partial match check could go here if needed
    # For now, strict subset check
    
    # If we are here, maybe it matches none?
    # Let's try to be a bit more lenient or check for specific unique columns
    if "auth_method" in cols: return "auth"
    if "dst_port" in cols and "proto" in cols: return "netflow"
    if "qtype" in cols: return "dns"
    if "method" in cols and "status" in cols: return "http"

    raise ValueError(f"Unknown log schema. Columns seen: {sorted(list(cols))[:50]}...")


# ----------------------------
# Small helpers
# ----------------------------
_IP_PRIVATE_RE = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.)")

def is_private_ip(ip: str) -> int:
    if not isinstance(ip, str):
        return 0
    return 1 if _IP_PRIVATE_RE.search(ip.strip()) else 0

def hour_dow(ts: str) -> Tuple[int, int]:
    try:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            return 0, 0
        return int(t.hour), int(t.dayofweek)
    except Exception:
        return 0, 0

def shannon_entropy(s: str) -> float:
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    probs = [s.count(ch) / len(s) for ch in set(s)]
    return float(-sum(p * math.log2(p) for p in probs))

def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Feature building (per row -> dict)
# Uses FeatureHasher so high-cardinality fields (IPs, paths, domains) are OK.
# ----------------------------
def row_to_features(kind: str, row: pd.Series) -> Dict[str, float]:
    f: Dict[str, float] = {}

    ts = row.get("timestamp", "")
    h, dow = hour_dow(ts)
    f["num:hour"] = float(h)
    f["num:dow"] = float(dow)

    if kind == "auth":
        src_ip = str(row.get("src_ip", ""))
        f["num:src_private"] = float(is_private_ip(src_ip))

        f[f"cat:app={row.get('app','')}"] = 1.0
        f[f"cat:dest_host={row.get('dest_host','')}"] = 1.0
        f[f"cat:auth_method={row.get('auth_method','')}"] = 1.0
        f[f"cat:outcome={row.get('outcome','')}"] = 1.0
        f[f"cat:fail_reason={row.get('fail_reason','')}"] = 1.0
        f[f"cat:src_region={row.get('src_region','')}"] = 1.0

        ua = str(row.get("user_agent", ""))
        ua_key = "OTHER"
        if "Chrome" in ua: ua_key = "CHROME"
        elif "Firefox" in ua: ua_key = "FIREFOX"
        elif "Safari" in ua and "Chrome" not in ua: ua_key = "SAFARI"
        elif "curl" in ua: ua_key = "CURL"
        elif "python-requests" in ua: ua_key = "PYTHON_REQUESTS"
        f[f"cat:user_agent_family={ua_key}"] = 1.0

    elif kind == "netflow":
        src_ip = str(row.get("src_ip", ""))
        dst_ip = str(row.get("dst_ip", ""))
        f["num:src_private"] = float(is_private_ip(src_ip))
        f["num:dst_private"] = float(is_private_ip(dst_ip))

        f[f"cat:proto={row.get('proto','')}"] = 1.0
        f[f"cat:dst_port={row.get('dst_port','')}"] = 1.0
        f[f"cat:tcp_flags={row.get('tcp_flags','')}"] = 1.0

        f["num:duration_ms"] = safe_float(row.get("duration_ms", 0))
        f["num:packets"] = safe_float(row.get("packets", 0))
        f["num:bytes"] = safe_float(row.get("bytes", 0))
        
        # ratios help
        pk = max(1.0, f["num:packets"])
        f["num:bytes_per_pkt"] = f["num:bytes"] / pk

    elif kind == "dns":
        src_ip = str(row.get("src_ip", ""))
        f["num:src_private"] = float(is_private_ip(src_ip))

        q = str(row.get("query", ""))
        f["num:q_len"] = float(len(q))
        f["num:q_entropy"] = float(shannon_entropy(q))
        f[f"cat:qtype={row.get('qtype','')}"] = 1.0
        f[f"cat:rcode={row.get('rcode','')}"] = 1.0
        f["num:answers"] = safe_float(row.get("answers", 0))
        f["num:ttl"] = safe_float(row.get("ttl", 0))
        f["num:is_nxdomain"] = 1.0 if str(row.get("rcode","")).upper() == "NXDOMAIN" else 0.0

        # crude TLD
        if "." in q:
            tld = q.rsplit(".", 1)[-1]
            f[f"cat:tld={tld}"] = 1.0

    elif kind == "http":
        src_ip = str(row.get("src_ip", ""))
        f["num:src_private"] = float(is_private_ip(src_ip))

        host = str(row.get("host", ""))
        path = str(row.get("path", ""))
        method = str(row.get("method", ""))
        status = safe_int(row.get("status", 0))

        f[f"cat:method={method}"] = 1.0
        f[f"cat:status={status}"] = 1.0

        # host + tld (hashed)
        f[f"cat:host={host}"] = 1.0
        if "." in host:
            tld = host.rsplit(".", 1)[-1]
            f[f"cat:tld={tld}"] = 1.0

        # tokenize path a bit (hashed)
        for tok in re.split(r"[/\?\=&\-\._]+", path.lower()):
            if tok:
                f[f"tok:path={tok}"] = 1.0

        f["num:bytes_in"] = safe_float(row.get("bytes_in", 0))
        f["num:bytes_out"] = safe_float(row.get("bytes_out", 0))

        ua = str(row.get("user_agent", ""))
        f[f"cat:ua_has_curl={int('curl' in ua)}"] = 1.0
        f[f"cat:ua_has_python={int('python-requests' in ua)}"] = 1.0

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return f

def build_X(kind: str, df: pd.DataFrame, n_features: int = 2**18):
    # FeatureHasher expects list[dict]
    dicts = [row_to_features(kind, df.iloc[i]) for i in range(len(df))]
    hasher = FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)
    X = hasher.transform(dicts)
    return X, hasher


# ----------------------------
# Training + bundle
# ----------------------------

@dataclass
class ModelInfo:
    kind: str
    classes_: List[str]
    precision: float
    recall: float
    false_positive_rate: float

@dataclass
class ModelBundle:
    # per-kind: model + hasher + info
    models: Dict[str, object]  # RandomForestClassifier or SGDClassifier
    hashers: Dict[str, FeatureHasher]
    infos: Dict[str, ModelInfo]

    def save(self, path: str):
        dump(self, path)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        return load(path)

def _make_target(df: pd.DataFrame) -> pd.Series:
    # multi-class target:
    # BENIGN for label==0, otherwise attack_type (fallback "SUSPICIOUS")
    label = df.get("label", pd.Series([0]*len(df)))
    attack_type = df.get("attack_type", pd.Series([""]*len(df)))

    y = []
    for i in range(len(df)):
        if int(label.iloc[i]) == 1:
            at = str(attack_type.iloc[i]).strip() or "SUSPICIOUS"
            y.append(at)
        else:
            y.append("BENIGN")
    return pd.Series(y)

def train_one(kind: str, df: pd.DataFrame) -> Tuple[object, FeatureHasher, ModelInfo]:
    df = df.copy()
    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    y = _make_target(df)
    X, hasher = build_X(kind, df, n_features=2**14)

    # Stratified split ensures both classes are represented
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # RandomForest with class_weight='balanced': automatically adjusts weights
    # inversely proportional to class frequencies. No additional sample_weight needed.
    try:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
    except Exception:
        # Fallback to SGD if RF fails for any reason
        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            max_iter=5000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)

    # Metrics (binary view: suspicious vs benign)
    y_pred = clf.predict(X_test)
    y_true_bin = (y_test != "BENIGN").astype(int)
    y_pred_bin = (pd.Series(y_pred) != "BENIGN").astype(int)

    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    
    # confusion_matrix logic needs to handle cases where not all labels are present
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = float(fp / max(1, (fp + tn)))

    info = ModelInfo(
        kind=kind,
        classes_=list(clf.classes_),
        precision=prec,
        recall=rec,
        false_positive_rate=fpr,
    )
    return clf, hasher, info

def train_bundle(paths: Dict[str, str], out_path: str = "model_bundle.joblib") -> ModelBundle:
    models, hashers, infos = {}, {}, {}

    for kind, p in paths.items():
        print(f"Training {kind} model from {p}...")
        try:
            df = pd.read_csv(p)
        except FileNotFoundError:
            print(f"  Warning: File {p} not found. Skipping {kind}.")
            continue

        detected = detect_log_kind(df)
        if detected != kind:
            print(f"  Warning: File {p} detected as {detected}, but expected {kind}. Proceeding anyway as {kind}...")
            # raise ValueError(f"File {p} detected as {detected}, but expected {kind}.")

        clf, hasher, info = train_one(kind, df)
        models[kind] = clf
        hashers[kind] = hasher
        infos[kind] = info
        print(f"  Finished {kind}: precision={info.precision:.2f}, recall={info.recall:.2f}")

    bundle = ModelBundle(models=models, hashers=hashers, infos=infos)
    bundle.save(out_path)
    return bundle


# ----------------------------
# Inference -> dashboard JSON
# ----------------------------

def _risk_level(suspicious_rate: float, mean_conf: float) -> str:
    # simple mapping; tune later
    score = 0.6 * suspicious_rate + 0.4 * mean_conf
    if score < 0.05:
        return "LOW"
    if score < 0.12:
        return "GUARDED"
    if score < 0.20:
        return "ELEVATED"
    return "CRITICAL"

def predict_dashboard(bundle: ModelBundle, df: pd.DataFrame, top_k: int = 50) -> Dict:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    kind = detect_log_kind(df)
    if kind not in bundle.models:
        raise ValueError(f"No model available for kind={kind}. Train it first.")

    clf = bundle.models[kind]
    hasher = bundle.hashers[kind]
    info = bundle.infos[kind]

    dicts = [row_to_features(kind, df.iloc[i]) for i in range(len(df))]
    X = hasher.transform(dicts)

    # predict proba -> confidence
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    pred_idx = np.argmax(proba, axis=1)
    pred_class = [classes[i] for i in pred_idx]
    conf = proba[np.arange(len(df)), pred_idx]

    # suspicious = predicted class != BENIGN
    is_susp = np.array([c != "BENIGN" for c in pred_class], dtype=bool)
    susp_count = int(is_susp.sum())
    total = int(len(df))

    mean_conf = float(np.mean(conf[is_susp])) if susp_count > 0 else 0.0
    suspicious_rate = float(susp_count / max(1, total))
    risk = _risk_level(suspicious_rate, mean_conf)

    # identified threats table
    # choose source_ip column if exists
    src_ip_col = "src_ip" if "src_ip" in df.columns else None

    # pick top-K suspicious by confidence
    idxs = np.argsort(-conf)
    threats = []
    for i in idxs:
        if not is_susp[i]:
            continue
        threats.append({
            "timestamp": str(df.iloc[i].get("timestamp","")),
            "event_type": str(pred_class[i]),
            "source_ip": str(df.iloc[i].get(src_ip_col,"")) if src_ip_col else "",
            "confidence": round(float(conf[i]) * 100, 1),
        })
        if len(threats) >= top_k:
            break

    # summarize “critical threats” as very high confidence
    critical = sum(1 for t in threats if t["confidence"] >= 95.0)

    return {
        "kind": kind,
        "total_logs_processed": total,
        "suspicious_activities": susp_count,
        "critical_threats": critical,
        "system_risk_level": risk,
        "identified_threats": threats,
        "ml_model_health": {
            "precision": round(info.precision * 100, 1),
            "recall": round(info.recall * 100, 1),
            "false_positive_rate": round(info.false_positive_rate * 100, 2),
            "classes": info.classes_,
        },
    }
