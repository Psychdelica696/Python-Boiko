# runtime-hook.py
import os
import sys


def setup_environment():
    """Налаштування середовища для PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        print("🔧 Налаштування середовища для EXE...")
        os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ['PATH']

        # Додаємо поточну директорію до шляху для бібліотек
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(current_dir)

        print("✅ Середовище налаштовано успішно")


# Викликаємо налаштування при імпорті
setup_environment()