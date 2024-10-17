import sys
import re
import torch
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QFileDialog, QMessageBox, QTabWidget,
    QComboBox, QListWidget, QListWidgetItem, QFormLayout, QTextBrowser
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os

class GPT2Generator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def generate_text(self, input_text, temperature_value, length_value, num_results, no_repeat_ngram_size):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=length_value,
            num_return_sequences=num_results,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=1.5,
            temperature=temperature_value,
            do_sample=True
        )

        result_text = ""
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            if i == 0:
                generated_text = generated_text.replace(input_text, "")

            lexer = get_lexer_by_name("python", stripall=True)
            formatter = HtmlFormatter(linenos=False, cssclass="code")
            highlighted_code = pygments.highlight(generated_text, lexer, formatter)
            generated_text = re.sub(r'(?<=[.])(?=[^\s])', '\n', generated_text)
            result_text += f"<style>{formatter.get_style_defs('.code')} .code {{ color: #000000; }}</style>"
            result_text += f"<pre>{highlighted_code}</pre>\n\n"

        return result_text

class GPT2App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_paths = []
        self.current_model_path = ""
        self.temperature_value = 0.1
        self.loadSettings()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Генератор ответов')
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(550, 300)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.createGenerateTab()
        self.createSettingsTab()

        self.statusBar = self.statusBar()
        self.show()

    def createGenerateTab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Генерация текста")

        layout = QFormLayout()

        self.input_text = QLineEdit()
        self.model_combo = QComboBox()
        self.updateModelComboBox()

        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setMinimum(1)
        self.temperature_slider.setMaximum(1000)
        self.temperature_slider.setValue(10)
        self.temperature_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temperature_slider.setTickInterval(10)
        self.temperature_display = QLabel('0.10')

        self.length_spinbox = QSpinBox()
        self.length_spinbox.setMinimum(10)
        self.length_spinbox.setMaximum(2048)
        self.length_spinbox.setValue(64)

        self.num_results_spinbox = QSpinBox()
        self.num_results_spinbox.setMinimum(1)
        self.num_results_spinbox.setMaximum(100)
        self.num_results_spinbox.setValue(1)

        self.generate_button = QPushButton('Сгенерировать ответ')

        self.result_text = QTextBrowser()

        layout.addRow(QLabel('Введите вопрос:'), self.input_text)
        layout.addRow(QLabel('Выберите модель:'), self.model_combo)
        layout.addRow(QLabel('Температура от 0.1 до 10.0:'), self.temperature_slider)
        layout.addRow(self.temperature_display)
        layout.addRow(QLabel('Длина текста до 2048 символов:'), self.length_spinbox)
        layout.addRow(QLabel('Количество вариантов от 1 до 100:'), self.num_results_spinbox)
        layout.addRow(self.generate_button)
        layout.addRow(QLabel('Результат:'), self.result_text)

        tab.setLayout(layout)

        self.generate_button.clicked.connect(self.generateText)
        self.temperature_slider.valueChanged.connect(self.updateTemperatureDisplay)
        self.model_combo.currentIndexChanged.connect(self.updateCurrentModelPath)

    def createSettingsTab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Настройки")

        layout = QVBoxLayout()

        self.model_list_label = QLabel("Сохраненные модели:")
        self.model_list_widget = QListWidget()

        self.add_model_button = QPushButton("Добавить модель")
        self.remove_model_button = QPushButton("Удалить выбранную модель")
        self.clear_settings_button = QPushButton("Очистить настройки")

        self.add_model_button.clicked.connect(self.addModel)
        self.remove_model_button.clicked.connect(self.removeModel)
        self.clear_settings_button.clicked.connect(self.clearSettings)

        layout.addWidget(self.model_list_label)
        layout.addWidget(self.model_list_widget)
        layout.addWidget(self.add_model_button)
        layout.addWidget(self.remove_model_button)
        layout.addWidget(self.clear_settings_button)

        tab.setLayout(layout)

        self.updateModelListWidget()

    def generateText(self):
        input_text = self.input_text.text()
        if not input_text:
            QMessageBox.warning(self, "Ошибка", "Введите текст для генерации.")
            return

        if not self.current_model_path:
            QMessageBox.warning(self, "Ошибка", "Выберите модель для генерации.")
            return

        length_value = self.length_spinbox.value()
        num_results = self.num_results_spinbox.value()
        no_repeat_ngram_size = 2

        try:
            self.statusBar.showMessage("Загрузка модели...")
            gpt2_generator = GPT2Generator(self.current_model_path)
            self.statusBar.showMessage("Генерация текста...")
            result_text = gpt2_generator.generate_text(
                input_text, self.temperature_value, length_value, num_results, no_repeat_ngram_size
            )
            self.result_text.setHtml(result_text)
            self.statusBar.showMessage("Генерация завершена.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
            self.statusBar.clearMessage()

    def updateTemperatureDisplay(self, value):
        self.temperature_value = value / 100.0
        self.temperature_display.setText(f'{self.temperature_value:.2f}')

    def updateModelComboBox(self):
        self.model_combo.clear()
        for model_path in self.model_paths:
            self.model_combo.addItem(model_path)

    def updateModelListWidget(self):
        self.model_list_widget.clear()
        for model_path in self.model_paths:
            self.model_list_widget.addItem(model_path)

    def updateCurrentModelPath(self):
        self.current_model_path = self.model_combo.currentText()

    def addModel(self):
        model_path = QFileDialog.getExistingDirectory(self, "Выберите директорию модели")
        if model_path:
            if model_path not in self.model_paths:
                self.model_paths.append(model_path)
                self.updateModelComboBox()
                self.updateModelListWidget()
                self.saveSettings()
            else:
                QMessageBox.warning(self, "Ошибка", "Модель уже добавлена.")

    def removeModel(self):
        selected_items = self.model_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Выберите модель для удаления.")
            return

        for item in selected_items:
            model_path = item.text()
            self.model_paths.remove(model_path)

        self.updateModelComboBox()
        self.updateModelListWidget()
        self.saveSettings()

    def clearSettings(self):
        reply = QMessageBox.question(self, "Подтверждение", "Вы уверены, что хотите очистить все настройки?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.model_paths = []
            self.current_model_path = ""
            self.updateModelComboBox()
            self.updateModelListWidget()
            self.saveSettings()

    def loadSettings(self):
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as file:
                settings = json.load(file)
                self.model_paths = settings.get("model_paths", [])
                self.current_model_path = settings.get("current_model_path", "")
                self.temperature_value = settings.get("temperature_value", 0.1)

    def saveSettings(self):
        settings = {
            "model_paths": self.model_paths,
            "current_model_path": self.current_model_path,
            "temperature_value": self.temperature_value
        }
        with open("settings.json", "w") as file:
            json.dump(settings, file)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GPT2App()
    sys.exit(app.exec())
