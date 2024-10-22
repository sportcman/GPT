from pydub import AudioSegment
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, 
    QSlider, QMessageBox, QTextEdit, QVBoxLayout, QWidget, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import os
import pygame

class TextToSpeechThread(QThread):
    finished = pyqtSignal(str)
    def __init__(self, text, language, pitch_shift):
        super().__init__()
        self.text = text
        self.language = language
        self.pitch_shift = pitch_shift

    def run(self):
        filename = "generated_speech.mp3"
        modified_filename = "modified_speech.mp3"
        try:
            tts = gTTS(text=self.text, lang=self.language)
            tts.save(filename)
            sound = AudioSegment.from_file(filename, format="mp3")
            new_sample_rate = int(sound.frame_rate * (2.0 ** (self.pitch_shift / 12.0)))
            sound_with_changed_pitch = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
            sound_with_changed_pitch = sound_with_changed_pitch.set_frame_rate(44100)
            sound_with_changed_pitch.export(modified_filename, format="mp3")
            pygame.mixer.init()
            pygame.mixer.music.load(modified_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            self.finished.emit("Аудио воспроизведено.")
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(modified_filename):
                os.remove(modified_filename)

class GPT2Generator:
    def __init__(self, model_path):
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
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            result_text += f"{generated_text}\n"
        return result_text

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT Текстовый генератор и озвучка")
        self.setGeometry(100, 100, 1000, 500)
        self.gpt2_generator = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Введите вопрос для генерации ответа...")
        layout.addWidget(QLabel("Вопрос:"))
        layout.addWidget(self.text_input)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.temp_slider.setMinimum(1)
        self.temp_slider.setMaximum(1000)  # Изменено: диапазон 1 - 1000 для высокой точности
        self.temp_slider.setValue(10)
        layout.addWidget(QLabel("Температура:"))
        layout.addWidget(self.temp_slider)

        self.temp_value_label = QLabel("0.03", self)  # Изменено: более точный диапазон
        layout.addWidget(self.temp_value_label)
        self.temp_slider.valueChanged.connect(self.update_temp_label)

        self.length_spinbox = QSpinBox(self)
        self.length_spinbox.setMinimum(10)
        self.length_spinbox.setMaximum(1024)
        self.length_spinbox.setValue(64)
        layout.addWidget(QLabel("Количество токенов:"))
        layout.addWidget(self.length_spinbox)

        self.model_button = QPushButton("Выбрать модель", self)
        layout.addWidget(self.model_button)
        self.model_button.clicked.connect(self.choose_model)

        self.generate_button = QPushButton("Сгенерировать ответ", self)
        layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.generate_text)

        self.result_text = QTextEdit(self)
        layout.addWidget(QLabel("Сгенерированный ответ:"))
        layout.addWidget(self.result_text)

        self.save_button = QPushButton("Сохранить в файл", self)
        layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_to_file)

        self.voice_combobox = QComboBox(self)
        self.voice_combobox.addItems(["ru", "en", "fr", "de", "es"])
        layout.addWidget(QLabel("Выберите голос:"))
        layout.addWidget(self.voice_combobox)

        self.pitch_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.pitch_slider.setMinimum(-16)
        self.pitch_slider.setMaximum(16)
        self.pitch_slider.setValue(0)
        layout.addWidget(QLabel("Тембр (голоса):"))
        layout.addWidget(self.pitch_slider)

        self.speech_button = QPushButton("Озвучить ответ", self)
        layout.addWidget(self.speech_button)
        self.speech_button.clicked.connect(self.convert_to_speech)

    def choose_model(self):
        model_path = QFileDialog.getExistingDirectory(self, "Выберите директорию модели")
        if model_path:
            self.gpt2_generator = GPT2Generator(model_path)

    def update_temp_label(self, value):
        temperature = 10 ** (- (1000 - value) / 100)  # Преобразование значения ползунка в диапазон от 0.000...1 до 10
        self.temp_value_label.setText(f"{temperature:.10f}")

    def generate_text(self):
        text = self.text_input.text()
        if not text or not self.gpt2_generator:
            QMessageBox.warning(self, "Ошибка", "Введите текст и выберите модель.")
            return

        temperature = 10 ** (- (1000 - self.temp_slider.value()) / 100)
        length_value = self.length_spinbox.value()

        try:
            generated_text = self.gpt2_generator.generate_text(
                text, temperature, length_value, 1, 2
            )
            self.result_text.setText(generated_text)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации текста: {str(e)}")

    def convert_to_speech(self):
        text = self.result_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Нет текста для озвучивания.")
            return

        selected_language = self.voice_combobox.currentText()
        pitch_shift = self.pitch_slider.value()
        self.tts_thread = TextToSpeechThread(text, selected_language, pitch_shift)
        self.tts_thread.finished.connect(self.show_message_box)
        self.tts_thread.start()

    def save_to_file(self):
        dialog = QFileDialog(self)
        dialog.setDefaultSuffix(".txt")
        dialog.setNameFilter("Text files (*.txt)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if dialog.exec():
            selected_file = dialog.selectedFiles()[0]
            with open(selected_file, 'w', encoding='utf-8') as file:
                file.write(self.result_text.toPlainText())
            QMessageBox.information(self, "Успех", "Файл успешно сохранён.")

    def show_message_box(self, message):
        QMessageBox.information(self, "Информация", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
