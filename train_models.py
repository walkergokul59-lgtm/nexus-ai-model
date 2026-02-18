# train_models.py
from modeling import train_bundle
import os

if __name__ == "__main__":
    # Define paths to the data files
    # These are expected to be in the same directory as this script
    paths = {
        "auth": "synthetic_auth_logs.csv",
        "netflow": "synthetic_netflow.csv",
        "dns": "synthetic_dns_logs.csv",
        "http": "synthetic_http_logs.csv",
    }
    
    # Check if files exist to avoid confusion
    missing_files = []
    for kind, p in paths.items():
        if not os.path.exists(p):
            missing_files.append(p)
    
    if missing_files:
        print("Error: The following data files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure you are running this script from the directory containing the data files.")
    else:
        print("Starting training process...")
        try:
            bundle = train_bundle(paths, out_path="model_bundle.joblib")
            print("\nSuccess! Saved model bundle to model_bundle.joblib")
            print("\nModel Info:")
            for k, info in bundle.infos.items():
                print(f"  {k.upper()}: Precision={info.precision:.2f}, Recall={info.recall:.2f}, FPR={info.false_positive_rate:.2f}")
        except Exception as e:
            print(f"\nAn error occurred during training: {e}")
            import traceback
            traceback.print_exc()
