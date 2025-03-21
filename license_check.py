from datetime import datetime, timedelta
import requests
import os
from PySide6.QtWidgets import QInputDialog, QMessageBox

LICENSE_DIR = os.path.expanduser("~/Library/Application Support/EyePrecision")
LICENSE_FILE = os.path.join(LICENSE_DIR, "license.txt")
GUMROAD_PRODUCT_ID = "T_Or5VnHnLezs6sbBNLVYA=="
MAX_USES = 5  # Your high test value, adjust later

def has_internet():
    try:
        requests.get("https://api.gumroad.com", timeout=5)
        return True
    except requests.RequestException:
        return False

def verify_license_online(key, increment=False):
    try:
        response = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": GUMROAD_PRODUCT_ID,
                "license_key": key,
                "increment_uses_count": "true" if increment else "false"
            },
            timeout=10
        )
        data = response.json()
        if data.get("success", False):
            uses = data.get("uses", 0)
            return True, uses
        return False, 0
    except requests.RequestException:
        return False, 0

def cache_license(key):
    os.makedirs(LICENSE_DIR, exist_ok=True)
    with open(LICENSE_FILE, "w") as f:
        f.write(f"{key}\n{datetime.now().isoformat()}")

def check_cached_license():
    if not os.path.exists(LICENSE_FILE):
        return False
    try:
        with open(LICENSE_FILE, "r") as f:
            lines = f.read().strip().splitlines()
            if len(lines) != 2:
                return False
            key, _ = lines
            return bool(key.strip())
    except (ValueError, IOError):
        return False

def prompt_for_license(parent=None):
    key, ok = QInputDialog.getText(
        parent,
        "License Activation",
        "Enter your EyePrecision license key (requires internet):"
    )
    if ok and key:
        key = key.replace(" ", "")
        # Check uses without incrementing first
        is_valid, uses = verify_license_online(key, increment=False)
        if not is_valid:
            QMessageBox.critical(parent, "Error", "Invalid license key. Contact chiropractorrobert@gmail.com for support.")
            return False
        if uses >= MAX_USES:  # Check before incrementing
            QMessageBox.critical(parent, "Error", f"This license key has been used too often (max {MAX_USES} activations). Contact chiropractorrobert@gmail.com.")
            return False
        # Now increment for this activation
        is_valid, uses = verify_license_online(key, increment=True)
        if is_valid and uses <= MAX_USES:  # Double-check post-increment
            cache_license(key)
            QMessageBox.information(parent, "Success", "License activated successfully!")
            return True
        else:
            QMessageBox.critical(parent, "Error", f"License activation failed (uses = {uses}, max = {MAX_USES}). Contact chiropractorrobert@gmail.com.")
            return False
    return False

def ensure_license(app):
    if check_cached_license():
        if has_internet():
            with open(LICENSE_FILE, "r") as f:
                cached_key = f.read().strip().splitlines()[0]
            is_valid, uses = verify_license_online(cached_key, increment=False)
            if is_valid:
                if uses > MAX_USES:
                    QMessageBox.critical(None, "License Error", f"This license key has been used too often (max {MAX_USES} activations). Contact chiropractorrobert@gmail.com.")
                    os.remove(LICENSE_FILE)
                    return prompt_for_license()
                cache_license(cached_key)  # Refresh timestamp
                return True
            else:
                os.remove(LICENSE_FILE)
                return prompt_for_license()
        else:
            return True
    else:
        if has_internet():
            return prompt_for_license()
        else:
            QMessageBox.critical(None, "No Internet", "Internet required for first activation.")
            return False