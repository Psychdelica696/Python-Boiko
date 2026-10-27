# app.py
import numpy as np
import joblib
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2


class LeukemiaDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩸 Застосунок для виявлення лейкемії")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        self.root.resizable(True, True)

        self.model = None
        self.image_path = None
        self.current_image = None

        print(f"🐍 Версія Python: {sys.version}")
        print(f"🧠 TensorFlow version: {tf.__version__}")

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        """Налаштування графічного інтерфейсу"""
        # Головний фрейм
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Заголовок
        title_label = tk.Label(
            main_frame,
            text="🩸 СИСТЕМА ВИЯВЛЕННЯ ЛЕЙКЕМІЇ",
            font=("Arial", 20, "bold"),
            fg="white",
            bg='#2c3e50'
        )
        title_label.pack(pady=15)

        # Фрейм для зображення
        image_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=3)
        image_frame.pack(pady=15, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            image_frame,
            text="Зображення не обрано\n\nНатисніть 'Завантажити зображення' щоб почати",
            bg='#34495e',
            fg='white',
            font=("Arial", 12),
            justify=tk.CENTER
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Фрейм керування
        controls_frame = tk.Frame(main_frame, bg='#2c3e50')
        controls_frame.pack(pady=15)

        # Кнопки
        btn_style = {
            'font': ("Arial", 11, "bold"),
            'width': 18,
            'height': 1,
            'cursor': 'hand2'
        }

        self.load_btn = tk.Button(
            controls_frame,
            text="📁 ЗАВАНТАЖИТИ ЗОБРАЖЕННЯ",
            command=self.load_image,
            **btn_style,
            bg='#3498db',
            fg='white',
            activebackground='#2980b9'
        )
        self.load_btn.grid(row=0, column=0, padx=10, pady=5)

        self.predict_btn = tk.Button(
            controls_frame,
            text="🔍 ПРОАНАЛІЗУВАТИ",
            command=self.predict,
            **btn_style,
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60'
        )
        self.predict_btn.grid(row=0, column=1, padx=10, pady=5)

        # Фрейм результатів
        results_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.SUNKEN, bd=2)
        results_frame.pack(pady=15, fill=tk.X, padx=20)

        self.result_label = tk.Label(
            results_frame,
            text="Ласкаво просимо до системи виявлення лейкемії!\n\n"
                 "1. Завантажте зображення клітин крові\n"
                 "2. Натисніть 'Проаналізувати' для отримання результату",
            font=("Arial", 11),
            bg='#34495e',
            fg='white',
            justify=tk.CENTER,
            wraplength=600
        )
        self.result_label.pack(pady=20, padx=20)

        # Індикатор прогресу
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            style='TProgressbar'
        )
        self.progress.pack(fill=tk.X, pady=10, padx=20)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("✅ Готово до роботи. Оберіть зображення...")

        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#34495e',
            fg='white',
            font=("Arial", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Налаштування стилів
        style = ttk.Style()
        style.configure('TProgressbar', thickness=15)

    def load_model(self):
        """Завантаження навченої моделі"""
        try:
            # Спершу пробуємо завантажити .joblib
            if os.path.exists('leukemia_model.joblib'):
                self.model = joblib.load('leukemia_model.joblib')
                self.status_var.set("✅ Модель успішно завантажена (joblib)")
                print("✅ Модель завантажена успішно з joblib")

            # Якщо joblib не знайдено, пробуємо .h5
            elif os.path.exists('leukemia_detection_model.h5'):
                self.model = keras.models.load_model('leukemia_detection_model.h5')
                self.status_var.set("✅ Модель успішно завантажена (h5)")
                print("✅ Модель завантажена успішно з h5")

            else:
                self.status_var.set("⚠️ Файли моделі не знайдено. Використовується тестовий режим")
                print("⚠️ Файли моделі не знайдено")

        except Exception as e:
            error_msg = f"❌ Помилка завантаження моделі: {str(e)}"
            self.status_var.set(error_msg)
            print(error_msg)
            messagebox.showerror("Помилка", error_msg)

    def load_image(self):
        """Завантаження зображення через діалог вибору файлу"""
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення клітин крові",
            filetypes=[
                ("Зображення", "*.bmp *.jpg *.jpeg *.png *.tiff *.tif"),
                ("Всі файли", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.result_label.config(
                text="✅ Зображення завантажено успішно!\n\nНатисніть 'Проаналізувати' для отримання результату")
            self.status_var.set(f"📁 Завантажено: {os.path.basename(file_path)}")

    def display_image(self, file_path):
        """Відображення обраного зображення"""
        try:
            img = Image.open(file_path)

            # Зменшуємо розмір для відображення
            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.current_image = img

        except Exception as e:
            error_msg = f"❌ Не вдалося завантажити зображення: {str(e)}"
            messagebox.showerror("Помилка", error_msg)
            self.status_var.set(error_msg)

    def preprocess_image(self, image_path):
        """Попередня обробка зображення для моделі"""
        try:
            # Завантажуємо та обробляємо зображення
            img = Image.open(image_path)
            img = img.convert('RGB')

            # Змінюємо розмір до 224x224
            img = img.resize((224, 224))

            # Конвертуємо в numpy array
            img_array = np.array(img) / 255.0  # Нормалізуємо
            img_array = np.expand_dims(img_array, axis=0)  # Додаємо batch dimension

            return img_array

        except Exception as e:
            print(f"Помилка обробки зображення: {e}")
            return None

    def predict(self):
        """Аналіз зображення на наявність лейкемії"""
        if not self.image_path:
            messagebox.showwarning("Увага", "Будь ласка, спочатку завантажте зображення!")
            return

        self.progress.start()
        self.status_var.set("🔍 Аналіз зображення...")
        self.result_label.config(text="Триває аналіз...\n\nБудь ласка, зачекайте")

        # Запускаємо аналіз в окремому потоці
        import threading
        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        """Поток для аналізу зображення"""
        try:
            # Обробляємо зображення
            processed_image = self.preprocess_image(self.image_path)

            if processed_image is None:
                raise Exception("Не вдалося обробити зображення")

            # Генеруємо результат аналізу
            if self.model:
                # Реальний аналіз (якщо модель завантажена)
                result = self._real_analysis(processed_image)
            else:
                # Тестовий аналіз
                result = self._test_analysis()

            # Оновлюємо інтерфейс в головному потоці
            self.root.after(0, self._update_results, result)

        except Exception as e:
            error_msg = f"❌ Помилка аналізу: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Помилка", error_msg))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.status_var.set("✅ Аналіз завершено"))

    def _real_analysis(self, processed_image):
        """Реальний аналіз з використанням моделі"""
        try:
            # Робимо прогноз
            prediction = self.model.predict(processed_image)
            confidence = float(prediction[0][0])

            # Визначаємо результат
            has_leukemia = confidence > 0.5

            return {
                'has_leukemia': has_leukemia,
                'confidence': confidence if has_leukemia else 1 - confidence,
                'score': confidence,
                'is_test': False
            }

        except Exception as e:
            print(f"Помилка прогнозу: {e}")
            # Повертаємо тестовий результат у разі помилки
            return self._test_analysis()

    def _test_analysis(self):
        """Тестовий аналіз (без моделі)"""
        confidence = random.uniform(0.6, 0.9)
        has_leukemia = random.choice([True, False])

        return {
            'has_leukemia': has_leukemia,
            'confidence': confidence,
            'score': random.uniform(0.1, 0.9),
            'is_test': True
        }

    def _update_results(self, result):
        """Оновлення інтерфейсу з результатами"""
        if result.get('is_test', False):
            result_text = "🧪 ТЕСТОВИЙ РЕЖИМ\n\n"
        else:
            result_text = ""

        if result['has_leukemia']:
            result_text += "❌ ВИЯВЛЕНО ЛЕЙКЕМІЮ\n\n"
            color = "red"
        else:
            result_text += "✅ ЛЕЙКЕМІЮ НЕ ВИЯВЛЕНО\n\n"
            color = "green"

        result_text += f"Впевненість: {result['confidence']:.1%}\n"
        result_text += f"Бал: {result['score']:.3f}"

        self.result_label.config(text=result_text, fg=color)

        # Пропонуємо показати детальний аналіз
        if messagebox.askyesno("Детальний аналіз", "Показати детальну візуалізацію результатів?"):
            self._show_detailed_analysis(result)

    def _show_detailed_analysis(self, result):
        """Показати детальну візуалізацію результатів"""
        plt.figure(figsize=(12, 5))
        plt.suptitle('📊 ДЕТАЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ', fontsize=16, fontweight='bold')

        # Графік ймовірностей
        plt.subplot(1, 2, 1)
        labels = ['Здоровий', 'Лейкемія']
        probabilities = [1 - result['score'], result['score']]
        colors = ['#2ecc71', '#e74c3c']

        bars = plt.bar(labels, probabilities, color=colors, alpha=0.7)
        plt.title('Ймовірності діагнозу', fontsize=12, fontweight='bold')
        plt.ylabel('Ймовірність', fontsize=11)
        plt.ylim(0, 1)

        # Додаємо значення на стовпцях
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')

        # Графік впевненості
        plt.subplot(1, 2, 2)
        confidence_data = [result['confidence'], 1 - result['confidence']]
        labels = ['Впевненість', 'Невпевненість']
        colors = ['#3498db', '#bdc3c7']

        plt.pie(confidence_data, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, wedgeprops={'alpha': 0.7})
        plt.title('Рівень впевненості аналізу', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()


def main():
    """Головна функція"""
    print("=" * 60)
    print("🩸 ЗАСТОСУНОК ДЛЯ ВИЯВЛЕННЯ ЛЕЙКЕМІЇ")
    print("=" * 60)

    root = tk.Tk()
    app = LeukemiaDetectionApp(root)

    # Центрування вікна
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()