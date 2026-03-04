"""
promote_model.py - Compare all versions and promote best to @champion alias.

Usage:
    python scripts/promote_model.py           # auto-promote best
    python scripts/promote_model.py --version 3   # promote specific version
    python scripts/promote_model.py --list        # list all versions
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI   = "sqlite:///mlflow.db"
MODEL_NAME     = "automl-fraud-detector"
CHAMPION_ALIAS = "champion"
METRIC_KEY     = "test_roc_auc"


def get_client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def get_all_versions(client):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    results  = []
    for v in versions:
        try:
            run     = client.get_run(v.run_id)
            metrics = run.data.metrics
            tags    = run.data.tags
        except Exception:
            metrics = {}
            tags    = {}
        results.append({
            "version":       v.version,
            "run_id":        v.run_id,
            "aliases":       v.aliases,
            "test_roc_auc":  metrics.get(METRIC_KEY, 0.0),
            "recall":        metrics.get("recall", 0.0),
            "f1_score":      metrics.get("f1_score", 0.0),
            "business_cost": metrics.get("business_cost", None),
            "model_name":    tags.get("best_model", "unknown"),
        })
    results.sort(key=lambda x: x["test_roc_auc"], reverse=True)
    return results


def list_versions(client):
    versions = get_all_versions(client)
    if not versions:
        print(f"No versions found for '{MODEL_NAME}'.")
        return
    print(f"\n{'─'*75}")
    print(f"  Model Registry: {MODEL_NAME}")
    print(f"{'─'*75}")
    print(f"  {'Ver':>3}  {'ROC-AUC':>8}  {'Recall':>7}  {'F1':>7}  {'Cost':>10}  {'Aliases'}")
    print(f"{'─'*75}")
    for v in versions:
        cost_str  = f"${v['business_cost']:,.0f}" if v["business_cost"] else "N/A"
        alias_str = ", ".join([f"@{a}" for a in (v["aliases"] or [])]) or "none"
        star      = " *" if CHAMPION_ALIAS in v["aliases"] else ""
        print(f"  v{v['version']:>2}  {v['test_roc_auc']:>8.5f}  "
              f"{v['recall']:>7.4f}  {v['f1_score']:>7.4f}  "
              f"{cost_str:>10}  {alias_str}{star}")
    print(f"{'─'*75}")
    champs = [v for v in versions if CHAMPION_ALIAS in (v["aliases"] or [])]    
    if champs:
        print(f"\n  Current @champion -> v{champs[0]['version']} (ROC-AUC: {champs[0]['test_roc_auc']:.5f})\n")
    else:
        print(f"\n  No @champion set yet.\n")


def set_champion(client, version):
    try:
        existing = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        print(f"  Removed @champion from v{existing.version}")
    except Exception:
        pass
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, str(version))
    client.set_model_version_tag(MODEL_NAME, str(version), "promoted_by", "promote_model.py")
    print(f"  v{version} is now @champion!")


def auto_promote(client):
    versions = get_all_versions(client)
    if not versions:
        print(f"No versions found for '{MODEL_NAME}'")
        return
    best = versions[0]
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        if str(current.version) == str(best["version"]):
            print(f"\n  v{best['version']} is already @champion (ROC-AUC: {best['test_roc_auc']:.5f})\n")
            return
        print(f"\n  Replacing v{current.version} with v{best['version']} as @champion...")
    except Exception:
        print(f"\n  Setting first @champion -> v{best['version']} (ROC-AUC: {best['test_roc_auc']:.5f})")
    set_champion(client, best["version"])


def main():
    parser = argparse.ArgumentParser(description="MLflow Registry Manager")
    parser.add_argument("--list",    action="store_true", help="List all versions")
    parser.add_argument("--version", type=int, default=None, help="Promote specific version")
    args = parser.parse_args()

    print(f"\n  Tracking: {TRACKING_URI} | Model: {MODEL_NAME}")
    client = get_client()

    if args.list:
        list_versions(client)
    elif args.version:
        set_champion(client, args.version)
        list_versions(client)
    else:
        auto_promote(client)
        list_versions(client)


if __name__ == "__main__":
    main()