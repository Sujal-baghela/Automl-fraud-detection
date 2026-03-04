"""
migrate_mlflow.py
-----------------
One-time migration from mlruns/ file store to SQLite backend.
Run this ONCE: python scripts/migrate_mlflow.py
"""

import os
import sys
import subprocess

def migrate():
    print("=" * 60)
    print("MLflow File Store → SQLite Migration")
    print("=" * 60)

    # Check if already migrated
    if os.path.exists("mlflow.db"):
        print("✅ mlflow.db already exists — skipping migration.")
        print("   SQLite backend is already set up.")
        return

    print("\n📦 Installing mlflow-migrate if needed...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlflow-migrate", "--quiet"], check=False)

    print("\n🔄 Migrating mlruns/ → mlflow.db ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlflow_migrate",
             "--from", "./mlruns",
             "--to",   "sqlite:///mlflow.db"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Migration successful!")
        else:
            print("⚠️  Migration tool failed — doing manual copy instead...")
            _manual_migrate()
    except Exception as e:
        print(f"⚠️  Migration tool not available ({e}) — doing manual copy...")
        _manual_migrate()

    print("\n✅ Done! From now on use: sqlite:///mlflow.db")
    print("   Run MLflow UI with:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


def _manual_migrate():
    """
    Fallback: initialize fresh SQLite DB and re-register models.
    Experiments/runs history stays in mlruns/ as read-only archive.
    New runs will go to mlflow.db.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    print("\n📋 Reading existing data from mlruns/...")
    mlflow.set_tracking_uri("mlruns")
    old_client = MlflowClient()
    old_versions = old_client.search_model_versions("name='automl-fraud-detector'")

    print(f"   Found {len(old_versions)} model versions")

    print("\n📝 Setting up fresh SQLite DB...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    new_client = MlflowClient()

    # Create experiment
    try:
        exp_id = new_client.create_experiment("fraud-detection")
        print(f"   Created experiment 'fraud-detection' (id={exp_id})")
    except Exception:
        print("   Experiment 'fraud-detection' already exists")

    print("\n✅ SQLite DB initialized.")
    print("   Note: Old run history stays in mlruns/ as archive.")
    print("   New training runs will be logged to mlflow.db")


if __name__ == "__main__":
    migrate()