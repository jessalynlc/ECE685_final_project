import os 
import os
from typing import Union, Dict, List

def list_tree(path: str, max_depth: int = 1) -> Union[Dict, List, str]:
    """
    Recursively build a nested dictionary representing the folder structure
    up to a specified depth. Intended for quickly inspecting repository layout
    without printing excessively deep trees.

    Args:
        path (str): Path to the root directory to inspect.
        max_depth (int, optional): Maximum depth to recurse into subdirectories.
                                   A value of 0 lists only files in the given
                                   directory. Defaults to 1.

    Returns:
        Union[Dict, List, str]: A nested dictionary describing the directory tree,
                                where folders map to subtrees and files appear in
                                a list under the '_files' key. '...' indicates
                                truncated depth.
    """
    def helper(current_path: str, depth: int) -> Union[Dict, str]:
        if depth > max_depth:
            return "..."

        tree: Dict[str, Union[Dict, List[str], str]] = {}

        try:
            for entry in os.listdir(current_path):
                full = os.path.join(current_path, entry)
                if os.path.isdir(full):
                    tree[entry] = helper(full, depth + 1)
                else:
                    # Ensure _files exists as a list of strings
                    if "_files" not in tree:
                        tree["_files"] = []
                    assert isinstance(tree["_files"], list)
                    tree["_files"].append(entry)
        except PermissionError:
            return "PERMISSION_DENIED"

        return tree

    return helper(path, 0)
