'''
libraries.py

This script automates the process of upgrading `pip` and installing a list of Python libraries. 
It uses the `subprocess` module to run `pip` commands and logs the entire process, including 
successes and failures, to both the console and a log file (`libraries.log`).

### Features:
- Upgrades `pip` to the latest version before installing any libraries.
- Installs a predefined list of Python libraries.
- Logs all operations to a log file and the console.
- Handles and logs errors if the installation fails.

### Requirements:
- Python 3.x installed on the system.
- Permissions to install packages using `pip`.

### Usage:
1. Customize the `libraries` list in the script with the packages you want to install.
2. Run the script from the terminal:
   
   ```bash
   python libraries.py
'''

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Logging level: INFO, can be set to DEBUG for more details
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("libraries.log"),  # Save logs to a file
        logging.StreamHandler(sys.stdout)  # Also print logs to the console
    ]
)

def run_pip_command(command):
    """
    Executes a pip command using subprocess and logs the result.

    Args:
        command (str): The pip command to be executed. This should be a string that 
                       corresponds to a valid pip command (e.g., 'install numpy', 
                       'install --upgrade pip').

    Raises:
        subprocess.CalledProcessError: If the pip command fails, this exception is raised and
                                       logged as an error. The exception is re-raised to signal failure.
    
    Example:
        To install a package using pip:
        >>> run_pip_command("install numpy")
        
        To upgrade pip itself:
        >>> run_pip_command("install --upgrade pip")
    """
    try:
        logging.info(f"Running pip command: {command}")
        subprocess.check_call([sys.executable, "-m", "pip"] + command.split())
        logging.info(f"Command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while executing command '{command}'. Error: {e}")
        raise


def upgrade_pip():
    """
    Upgrades the pip package manager to the latest version.

    This function runs the `pip install --upgrade pip` command to ensure that 
    the latest version of pip is being used before installing other libraries.
    
    Logs:
        - Logs the beginning and successful completion of the pip upgrade process.
        - Logs an error if the upgrade fails.

    Raises:
        Exception: If the pip upgrade fails, an error is logged and the exception is raised.
    
    Example:
        To upgrade pip before installing other libraries:
        >>> upgrade_pip()
    """
    try:
        logging.info("Upgrading pip...")
        run_pip_command("install --upgrade pip")
        logging.info("pip upgraded successfully.")
    except Exception as e:
        logging.error(f"Failed to upgrade pip: {e}")
        raise


def install_libraries(libraries):
    """
    Installs a list of Python libraries using pip.

    Args:
        libraries (list): A list of strings where each string is the name of a library to install.
                          For example, ['numpy', 'pandas', 'scikit-learn'].

    Logs:
        - Logs the start of each library installation.
        - Logs the successful installation of each library.
        - Logs an error if a library installation fails.

    Example:
        To install a list of libraries:
        >>> install_libraries(['numpy', 'pandas', 'scikit-learn'])
    """
    for library in libraries:
        try:
            logging.info(f"Installing library {library}...")
            run_pip_command(f"install {library}")
            logging.info(f"{library} installed successfully.")
        except Exception as e:
            logging.error(f"Failed to install {library}: {e}")


# List of libraries to install
libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "scikit-learn",
    "openpyxl",
    "shap"
]
upgrade_pip()
install_libraries(libraries)
logging.info("Installation of all libraries completed!")
