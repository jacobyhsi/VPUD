import sys
import os

def modify_sys_path():
    """
    Run this function to add the src directory to sys.path.
    """
    # Get the absolute path of the project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # Add the src directory to sys.path
    sys.path.append(project_root)

    return

def get_src_dir_path(path: str = None):
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    if path is None:
        return project_root
    else:
        return os.path.join(project_root, path)