
import shutil
from pathlib import Path


def download_dataset(target_dir: Path | None = None) -> Path:
    import kagglehub
    
    print("Downloading Poland Vehicle License Plate Dataset...")
    print("This may take a few minutes on first run.\n")
    
    # Download dataset using kagglehub
    dataset_path = kagglehub.dataset_download(
        "piotrstefaskiue/poland-vehicle-license-plate-dataset"
    )
    
    dataset_path = Path(dataset_path)
    print(f"\nDataset downloaded to: {dataset_path}")
    
    # If target directory specified, copy there
    if target_dir:
        target_dir = Path(target_dir)
        if target_dir != dataset_path:
            print(f"Copying to: {target_dir}")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(dataset_path, target_dir)
            dataset_path = target_dir
    
    # List contents
    print("\nDataset structure:")
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            subfiles = list(item.iterdir())
            print(f"  {item.name}/ ({len(subfiles)} items)")
        else:
            print(f"  {item.name}")
    
    return dataset_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download license plate dataset")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Target directory for dataset (default: kagglehub cache)"
    )
    
    args = parser.parse_args()
    
    # Check if kagglehub is available
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("Error: kagglehub not installed. Run: uv pip install kagglehub")
        return 1
    
    try:
        dataset_path = download_dataset(args.target_dir)
        print(f"\n✓ Dataset ready at: {dataset_path}")
        return 0
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
