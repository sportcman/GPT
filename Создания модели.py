import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget, QFormLayout, QTabWidget, QProgressBar, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

class ModelThread(QThread):
    update_text_signal = pyqtSignal(str)
    model_saved_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            if not os.path.exists(self.config['tokenizer_path']):
                self.update_text_signal.emit(f"Путь к токенизатору не существует: {self.config['tokenizer_path']}")
                return
            
            self.update_text_signal.emit("Загружаем предварительно созданный токенизатор")
            tokenizer = GPT2Tokenizer.from_pretrained(self.config['tokenizer_path'])

            self.update_text_signal.emit("Создаем конфигурацию модели с нужными параметрами")
            model_config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_layer=self.config['n_layer'],
                n_head=self.config['n_head'],
                n_embd=self.config['n_embd'],
                intermediate_size=self.config['intermediate_size'],
                hidden_size=self.config['hidden_size'],
                max_position_embeddings=self.config['max_position_embeddings'],
                num_attention_heads=self.config['num_attention_heads'],
                gradient_checkpointing=self.config['gradient_checkpointing'],
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                sep_token_id=tokenizer.sep_token_id,
                activation_function=self.config['activation_function'],
                initializer_range=self.config['initializer_range'],
                layer_norm_eps=self.config['layer_norm_eps'],
                scale_attn_by_inverse_layer_idx=self.config['scale_attn_by_inverse_layer_idx'],
                reorder_and_upcast_attn=self.config['reorder_and_upcast_attn']
            )

            self.update_text_signal.emit("Создаем модель на основе заданной конфигурации")
            model = GPT2LMHeadModel(config=model_config)
            model.resize_token_embeddings(len(tokenizer))
            self.update_text_signal.emit("Модель создана.")

            # Прогресс сохранения модели
            self.progress_signal.emit(50)
            
            self.update_text_signal.emit("Сохраняем модель и токенизатор.")
            model.save_pretrained(self.config['model_save_path'])
            tokenizer.save_pretrained(self.config['model_save_path'])
            self.update_text_signal.emit("Модель и токенизатор сохранены.")
            
            # Прогресс завершен
            self.progress_signal.emit(100)
            self.model_saved_signal.emit()
        except Exception as e:
            self.update_text_signal.emit(f"Ошибка: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Создание модели GPT")
        self.setGeometry(100, 100, 600, 500)
        
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.createModelTab()
        self.createSettingsTab()

        self.model_thread = None

    def createModelTab(self):
        self.modelTab = QWidget()
        self.tabs.addTab(self.modelTab, "Создание модели")

        layout = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        self.exit_button = QPushButton("Выход", self)
        self.exit_button.setEnabled(False)
        self.exit_button.clicked.connect(self.close)

        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.exit_button)
        
        self.modelTab.setLayout(layout)

    def createSettingsTab(self):
        self.settingsTab = QWidget()
        self.tabs.addTab(self.settingsTab, "Настройки")

        layout = QFormLayout()

        self.tokenizerPathEdit = QLineEdit("F:/GPT_pro/token")
        self.modelSavePathEdit = QLineEdit("F:/GPT_pro/model")
        self.nLayerEdit = QLineEdit("16")
        self.nHeadEdit = QLineEdit("256")
        self.nEmbdEdit = QLineEdit("1024")
        self.intermediateSizeEdit = QLineEdit("1024")
        self.hiddenSizeEdit = QLineEdit("512")
        self.maxPositionEmbeddingsEdit = QLineEdit("1024")
        self.numAttentionHeadsEdit = QLineEdit("256")
        self.gradientCheckpointingEdit = QLineEdit("True")
        self.activationFunctionEdit = QLineEdit("gelu_new")
        self.initializerRangeEdit = QLineEdit("0.02")
        self.layerNormEpsEdit = QLineEdit("1e-5")
        self.scaleAttnByInverseLayerIdxEdit = QLineEdit("True")
        self.reorderAndUpcastAttnEdit = QLineEdit("True")

        layout.addRow("Путь к токенизатору:", self.tokenizerPathEdit)
        layout.addRow("Путь сохранения модели:", self.modelSavePathEdit)
        layout.addRow("Количество слоев:", self.nLayerEdit)
        layout.addRow("Количество голов:", self.nHeadEdit)
        layout.addRow("Размер эмбеддингов:", self.nEmbdEdit)
        layout.addRow("Промежуточный размер:", self.intermediateSizeEdit)
        layout.addRow("Скрытый размер:", self.hiddenSizeEdit)
        layout.addRow("Макс. позиционные эмбеддинги:", self.maxPositionEmbeddingsEdit)
        layout.addRow("Количество голов внимания:", self.numAttentionHeadsEdit)
        layout.addRow("Контроль градиентов:", self.gradientCheckpointingEdit)
        layout.addRow("Функция активации:", self.activationFunctionEdit)
        layout.addRow("Диапазон инициализации:", self.initializerRangeEdit)
        layout.addRow("Эпсилон нормализации слоев:", self.layerNormEpsEdit)
        layout.addRow("Масштабировать внимание по индексу слоя:", self.scaleAttnByInverseLayerIdxEdit)
        layout.addRow("Перестроить и увеличивать внимание:", self.reorderAndUpcastAttnEdit)

        self.start_button = QPushButton("Начать создание модели")
        self.start_button.clicked.connect(self.start_model_creation)

        layout.addWidget(self.start_button)
        
        self.settingsTab.setLayout(layout)

    def start_model_creation(self):
        try:
            config = {
                'tokenizer_path': self.tokenizerPathEdit.text(),
                'model_save_path': self.modelSavePathEdit.text(),
                'n_layer': int(self.nLayerEdit.text()),
                'n_head': int(self.nHeadEdit.text()),
                'n_embd': int(self.nEmbdEdit.text()),
                'intermediate_size': int(self.intermediateSizeEdit.text()),
                'hidden_size': int(self.hiddenSizeEdit.text()),
                'max_position_embeddings': int(self.maxPositionEmbeddingsEdit.text()),
                'num_attention_heads': int(self.numAttentionHeadsEdit.text()),
                'gradient_checkpointing': self.gradientCheckpointingEdit.text().lower() == 'true',
                'activation_function': self.activationFunctionEdit.text(),
                'initializer_range': float(self.initializerRangeEdit.text()),
                'layer_norm_eps': float(self.layerNormEpsEdit.text()),
                'scale_attn_by_inverse_layer_idx': self.scaleAttnByInverseLayerIdxEdit.text().lower() == 'true',
                'reorder_and_upcast_attn': self.reorderAndUpcastAttnEdit.text().lower() == 'true'
            }

            self.model_thread = ModelThread(config)
            self.model_thread.update_text_signal.connect(self.update_text)
            self.model_thread.model_saved_signal.connect(self.enable_exit_button)
            self.model_thread.progress_signal.connect(self.update_progress)
            self.model_thread.start()

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректные значения параметров.")

    def update_text(self, text):
        self.text_edit.append(text)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def enable_exit_button(self):
        self.exit_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
