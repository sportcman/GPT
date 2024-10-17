import os
import shutil
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QProgressBar, QTextEdit, QLabel, QFileDialog, QApplication, QSpinBox, QFormLayout, QCheckBox
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR

# Кастомный датасет с опцией перекрытия фрагментов
class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size, overlap_size=0):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.encode(text)
        step_size = block_size - overlap_size if overlap_size > 0 else block_size
        
        for i in range(0, len(tokenized_text) - block_size + 1, step_size):
            self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class TrainingWorker(QThread):
    update_progress = pyqtSignal(int)
    update_log = pyqtSignal(str)
    training_finished = pyqtSignal()

    def __init__(self, model_path, dataset_path, device, batch_size, epochs, gradient_accumulation_steps, overlap_enabled, overlap_size, block_size):
        super().__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.overlap_enabled = overlap_enabled
        self.overlap_size = overlap_size
        self.block_size = block_size
        self.is_running = True

    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.update_log.emit(f"Модель не найдена по указанному пути: {self.model_path}")
                self.training_finished.emit()
                return

            if not os.path.exists(os.path.join(self.model_path, 'tokenizer_config.json')):
                self.update_log.emit(f"Токенизатор не найден по указанному пути: {self.model_path}")
                self.training_finished.emit()
                return

            tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            model.to(self.device)

            overlap_size = self.overlap_size if self.overlap_enabled else 0

            dataset = CustomTextDataset(tokenizer, self.dataset_path, block_size=self.block_size, overlap_size=overlap_size)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            optimizer = AdamW(model.parameters(), lr=5e-5)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

            model.train()

            total_steps = len(data_loader) * self.epochs
            accum_steps = 0

            for epoch in range(self.epochs):
                if not self.is_running:
                    break

                for i, batch in enumerate(data_loader):
                    if not self.is_running:
                        break

                    inputs = batch.to(self.device)
                    labels = batch.to(self.device)

                    outputs = model(inputs, labels=labels, attention_mask=inputs != tokenizer.pad_token_id)
                    loss = outputs.loss

                    loss.backward()

                    if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_steps = 0
                    else:
                        accum_steps += 1

                    current_step = epoch * len(data_loader) + i + 1
                    self.update_progress.emit(int((current_step / total_steps) * 100))
                    self.update_log.emit(f"Эпоха {epoch+1}/{self.epochs}, Партия {i+1}/{len(data_loader)}, Потеря: {loss.item()}")

                scheduler.step()
                self.update_log.emit(f"Эпоха {epoch+1}/{self.epochs} завершена.")

            if self.is_running:
                self.save_model_and_tokenizer(model, tokenizer)
            else:
                self.update_log.emit("Обучение остановлено пользователем.")

            self.training_finished.emit()

        except Exception as e:
            self.update_log.emit(f"Произошла ошибка: {str(e)}")
            self.training_finished.emit()

    def stop_training(self):
        self.is_running = False
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        model = GPT2LMHeadModel.from_pretrained(self.model_path)
        model.to(self.device)
        self.save_model_and_tokenizer(model, tokenizer)
        self.update_log.emit("Обучение было остановлено, модель сохранена.")

    def save_model_and_tokenizer(self, model, tokenizer):
        model_save_path = os.path.join(self.model_path, '')
        tokenizer_save_path = os.path.join(self.model_path, '')

        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path)
        if os.path.exists(tokenizer_save_path):
            shutil.rmtree(tokenizer_save_path)

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(tokenizer_save_path, exist_ok=True)

        try:
            self.update_log.emit(f"Сохранение модели в {model_save_path}")
            model.save_pretrained(model_save_path)

            self.update_log.emit(f"Сохранение токенизатора в {tokenizer_save_path}")
            tokenizer.save_pretrained(tokenizer_save_path)

            self.update_log.emit("Модель и токенизатор успешно сохранены.")
        except Exception as e:
            self.update_log.emit(f"Ошибка при сохранении модели или токенизатора: {str(e)}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.start_button = QPushButton('Начать обучение', self)
        self.start_button.clicked.connect(self.start_training)

        self.stop_button = QPushButton('Остановить обучение', self)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)

        self.model_label = QLabel('Модель: Не выбрана', self)
        self.dataset_label = QLabel('Датасет: Не выбран', self)

        self.model_button = QPushButton('Выбрать модель', self)
        self.model_button.clicked.connect(self.select_model)

        self.dataset_button = QPushButton('Выбрать датасет', self)
        self.dataset_button.clicked.connect(self.select_dataset)

        # Поля для ввода параметров batch_size, epochs, gradient_accumulation_steps, block_size
        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(8)

        self.epochs_input = QSpinBox(self)
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(1)

        self.gradient_accumulation_input = QSpinBox(self)
        self.gradient_accumulation_input.setRange(1, 1024)
        self.gradient_accumulation_input.setValue(16)

        self.block_size_input = QSpinBox(self)
        self.block_size_input.setRange(1, 1024)
        self.block_size_input.setValue(128)

        # Флажок для включения перекрытия фрагментов
        self.overlap_checkbox = QCheckBox("Включить перекрытие фрагментов", self)
        self.overlap_size_input = QSpinBox(self)
        self.overlap_size_input.setRange(1, 1024)
        self.overlap_size_input.setValue(64)

        params_form = QFormLayout()
        params_form.addRow("Batch Size", self.batch_size_input)
        params_form.addRow("Epochs", self.epochs_input)
        params_form.addRow("Gradient Accumulation Steps", self.gradient_accumulation_input)
        params_form.addRow("Block Size", self.block_size_input)  # Добавлено поле для block_size
        params_form.addRow(self.overlap_checkbox)
        params_form.addRow("Размер перекрытия", self.overlap_size_input)

        layout.addWidget(self.model_button)
        layout.addWidget(self.model_label)
        layout.addWidget(self.dataset_button)
        layout.addWidget(self.dataset_label)
        layout.addLayout(params_form)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)

        self.setLayout(layout)
        self.setWindowTitle('Обучение модели')

    def select_model(self):
        model_path = QFileDialog.getExistingDirectory(self, "Выбрать директорию модели")
        if model_path:
            self.model_label.setText(f'Модель: {model_path}')
            self.model_path = model_path

    def select_dataset(self):
        dataset_path = QFileDialog.getOpenFileName(self, "Выбрать файл датасета", "", "Text Files (*.txt);;All Files (*)")[0]
        if dataset_path:
            self.dataset_label.setText(f'Датасет: {dataset_path}')
            self.dataset_path = dataset_path

    def start_training(self):
        self.log_text.clear()

        batch_size = self.batch_size_input.value()
        epochs = self.epochs_input.value()
        gradient_accumulation_steps = self.gradient_accumulation_input.value()
        overlap_enabled = self.overlap_checkbox.isChecked()
        overlap_size = self.overlap_size_input.value()
        block_size = self.block_size_input.value()  # Получаем значение block_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.worker = TrainingWorker(self.model_path, self.dataset_path, device, batch_size, epochs, gradient_accumulation_steps, overlap_enabled, overlap_size, block_size)  # Передаем block_size в TrainingWorker
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.update_log.connect(self.log_text.append)
        self.worker.training_finished.connect(self.on_training_finished)

        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_training(self):
        if self.worker:
            self.worker.stop_training()

    def on_training_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append("Обучение завершено.")


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
