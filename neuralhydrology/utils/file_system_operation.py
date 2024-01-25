from pathlib import Path
from typing import Union, Optional


def get_latest_touched_directory(parent_path: Union[str, Path]) -> Optional[Path]:
    """
    Get the path of the most recently modified directory within a given parent path.

    Args:
        parent_path (Union[str, Path]): The path to the parent directory.

    Returns:
        Optional[Path]: The path of the most recently modified directory, or None if no directories are found.

    Raises:
        FileNotFoundError: If the specified parent path does not exist.
    """
    parent_path = Path(parent_path)

    # Check if the parent path exists
    if not parent_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {parent_path}")

    # Get all directories in the parent path
    directories: list[Path] = [d for d in parent_path.iterdir() if d.is_dir()]

    # Sort directories based on modification time (most recently modified first)
    sorted_directories: list[Path] = sorted(directories, key=lambda d: d.stat().st_mtime, reverse=True)

    # Return the path of the most recently modified directory
    if sorted_directories:
        return sorted_directories[0]
    else:
        return None


# Example usage:
if __name__ == "__main__":
    parent_directory: str = "/path/to/parent/directory"
    latest_directory: Optional[Path] = get_latest_touched_directory(parent_directory)

    if latest_directory:
        print("Latest touched directory:", latest_directory)
    else:
        print("No directories found.")
