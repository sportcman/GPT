import sys
import os
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit, QLabel, QFormLayout, QStatusBar, QLineEdit
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class TokenizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tokenizer Creator")
        self.setGeometry(700, 300, 800, 600)  # Увеличено для больших текстовых полей

        self.default_tokens = [
            # Специальные токены
            '<unk>', '<s>', '</s>', '<pad>', '<mask>',
    
            # Пробелы и пунктуация
            ' ', ',', '.', '!', '—', '-', ';', ':', '(', ')', '"', '?', '«', '»', "'", '[', ']',
            '{', '}', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '~', '`', '<', '>', '„',
    
            # Кириллица (маленькие буквы)
            'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п',
            'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
    
            # Кириллица (большие буквы)
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П',
            'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
    
            # Латиница (маленькие буквы)
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    
            # Латиница (большие буквы)
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    
            # Цифры
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

            # Математические и логические символы
            '≠', '≈', '±', '√', '∞', '∫', '∑', '∏', '∂', '∇', '→', '←', '⇔', '∀', '∃', '∧', '∨', '⊥', '⊂', '⊃',
    
            # Символы валют
            '₽', '$', '€', '£', '¥', '₴', '₹', '₩', '฿', '₿', '¢', '৳',
            
            # Прочие символы
            '°', '©', '®', '™', '§', '¶',

            #  Диакритические знаки (латинские буквы с акцентами)
            'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ä', 'ö', 'ß', 'ç',

            # Русские слоги
            'ба', 'бо', 'бу', 'бы', 'бе', 'би', 'бё', 'бю', 'бя',
            'ва', 'во', 'ву', 'вы', 'ве', 'ви', 'вё', 'вю', 'вя',
            'га', 'го', 'гу', 'гы', 'ге', 'ги', 'гё', 'гю', 'гя',
            'да', 'до', 'ду', 'ды', 'де', 'ди', 'дё', 'дю', 'дя',
            'жа', 'жо', 'жу', 'жы', 'же', 'жи', 'жё', 'жю', 'жя',
            'за', 'зо', 'зу', 'зы', 'зе', 'зи', 'зё', 'зю', 'зя',
            'ка', 'ко', 'ку', 'кы', 'ке', 'ки', 'кё', 'кю', 'кя',
            'ла', 'ло', 'лу', 'лы', 'ле', 'ли', 'лё', 'лю', 'ля',
            'ма', 'мо', 'му', 'мы', 'ме', 'ми', 'мё', 'мю', 'мя',
            'на', 'но', 'ну', 'ны', 'не', 'ни', 'нё', 'ню', 'ня',
            'па', 'по', 'пу', 'пы', 'пе', 'пи', 'пё', 'пю', 'пя',
            'ра', 'ро', 'ру', 'ры', 'ре', 'ри', 'рё', 'рю', 'ря',
            'са', 'со', 'су', 'сы', 'се', 'си', 'сё', 'сю', 'ся',
            'та', 'то', 'ту', 'ты', 'те', 'ти', 'тё', 'тю', 'тя',
            'фа', 'фо', 'фу', 'фы', 'фе', 'фи', 'фё', 'фю', 'фя',
            'ха', 'хо', 'ху', 'хы', 'хе', 'хи', 'хё', 'хю', 'хя',
            'ца', 'цо', 'цу', 'цы', 'це', 'ци', 'цё', 'цю', 'ця',
            'ча', 'чо', 'чу', 'чы', 'че', 'чи', 'чё', 'чю', 'чя',
            'ша', 'шо', 'шу', 'шы', 'ше', 'ши', 'шё', 'шю', 'шя',
            'ща', 'що', 'щу', 'щы', 'ще', 'щи', 'щё', 'щю', 'щя',
            'эа', 'эо', 'эу', 'эы', 'эе', 'эи', 'эё', 'эю', 'эя',
            'юа', 'юо', 'юу', 'юы', 'юе', 'юи', 'юё', 'юю', 'юя',
            'яа', 'яо', 'яу', 'яы', 'яе', 'яи', 'яё', 'яю', 'яя',
            'аба', 'або', 'абу', 'абы', 'абе', 'аби', 'абё', 'абю', 'абя',
            'ава', 'аво', 'аву', 'авы', 'аве', 'ави', 'авё', 'авю', 'авя',
            'ага', 'аго', 'агу', 'агы', 'аге', 'аги', 'агё', 'агю', 'агя',
            'ада', 'адо', 'аду', 'ады', 'аде', 'ади', 'адё', 'адю', 'адя',
            'ажа', 'ажо', 'ажу', 'ажы', 'аже', 'ажи', 'ажё', 'ажю', 'ажя',
            'аза', 'азо', 'азу', 'азы', 'азе', 'ази', 'азё', 'азю', 'азя',
            'ака', 'ако', 'аку', 'акы', 'аке', 'аки', 'акё', 'акю', 'акя',
            'ала', 'ало', 'алу', 'алы', 'але', 'али', 'алё', 'алю', 'аля',
            'ама', 'амо', 'аму', 'амы', 'аме', 'ами', 'амё', 'амю', 'амя',
            'ана', 'ано', 'ану', 'аны', 'ане', 'ани', 'анё', 'аню', 'аня',
            'апа', 'апо', 'апу', 'апы', 'апе', 'апи', 'апё', 'апю', 'апя',
            'ара', 'аро', 'ару', 'ары', 'аре', 'ари', 'арё', 'арю', 'аря',
            'аса', 'асо', 'асу', 'асы', 'асе', 'аси', 'асё', 'асю', 'ася',
            'ата', 'ато', 'ату', 'аты', 'ате', 'ати', 'атё', 'атю', 'атя',
            'афа', 'афо', 'афу', 'афы', 'афе', 'афи', 'афё', 'афю', 'афя',
            'аха', 'ахо', 'аху', 'ахы', 'ахе', 'ахи', 'ахё', 'ахю', 'ахя',
            'аца', 'ацо', 'ацу', 'ацы', 'аце', 'аци', 'ацё', 'ацю', 'аця',
            'ача', 'ачо', 'ачу', 'ачи', 'ачё', 'ачю', 'ачя',
            'аша', 'ашо', 'ашу', 'аши', 'ашё', 'ашю', 'ашя',
            'аща', 'ащо', 'ащу', 'ащи', 'ащё', 'ащю', 'ащя',
            'аэ', 'аю', 'ая',

            # англиские слоги
            'ba', 'be', 'bi', 'bo', 'bu', 'by', 'bae', 'bee', 'boo', 'bay', 'biu', 'bow', 'bai', 'bao',
            'ca', 'ce', 'ci', 'co', 'cu', 'cy', 'cow', 'cue',
            'da', 'de', 'di', 'do', 'du', 'dy', 'day', 'dew', 'dye',
            'fa', 'fe', 'fi', 'fo', 'fu', 'fy', 'few',
            'ga', 'ge', 'gi', 'go', 'gu', 'gy', 'guy', 'goo', 'gay', 'gow',
            'ha', 'he', 'hi', 'ho', 'hu', 'hy', 'hey', 'how', 'hew', 'hay',
            'ja', 'je', 'ji', 'jo', 'ju', 'jay', 'jaw', 'joy',
            'ka', 'ke', 'ki', 'ko', 'ku', 'kay', 'key', 'kow',
            'la', 'le', 'li', 'lo', 'lu', 'ly', 'lay', 'low', 'lee', 'lie',
            'ma', 'me', 'mi', 'mo', 'mu', 'my', 'may', 'mew', 'moo',
            'na', 'ne', 'ni', 'no', 'nu', 'ny', 'nay', 'new', 'now', 'nee', 'nigh',
            'pa', 'pe', 'pi', 'po', 'pu', 'py', 'paw', 'pay', 'pee', 'pie',
            'ra', 're', 'ri', 'ro', 'ru', 'ry', 'raw', 'ray', 'row', 'rie', 'rye',
            'sa', 'se', 'si', 'so', 'su', 'sy', 'saw', 'say', 'see', 'sow',
            'ta', 'te', 'ti', 'to', 'tu', 'ty', 'two', 'tie', 'tow', 'taw',
            'va', 've', 'vi', 'vo', 'vu', 'vy', 'vow', 'vie',
            'wa', 'we', 'wi', 'wo', 'wu', 'wy', 'wow', 'why', 'wee', 'wye',
            'ya', 'ye', 'yi', 'yo', 'yu', 'yaw', 'yay', 'you', 'yew',
            'za', 'ze', 'zi', 'zo', 'zu', 'zy', 'zoo', 'zow',
            'al', 'am', 'an', 'at', 'el', 'en', 'ex', 'in', 'is', 'on', 'or', 'up', 'us'
            ]

        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Установка шрифта
        font = QFont("Segoe UI", 12)
        self.setFont(font)

        # Input for tokens
        self.tokens_text_edit = QTextEdit()
        self.tokens_text_edit.setPlaceholderText("Введите токены через запятую. По умолчанию: <unk>, <s>, </s>, <pad>, <mask>")
        self.tokens_text_edit.setText(', '.join(self.default_tokens))  # Установка токенов по умолчанию
        self.tokens_text_edit.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.tokens_text_edit.setFont(QFont("Segoe UI", 14))

        # Output directory selection
        self.output_dir_line_edit = QLineEdit()
        self.output_dir_line_edit.setPlaceholderText("Выберите папку для сохранения")
        self.output_dir_line_edit.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.output_dir_line_edit.setFont(QFont("Segoe UI", 14))

        self.output_dir_button = QPushButton("Выбрать папку")
        self.output_dir_button.setStyleSheet("background-color: #0078D4; color: #FFFFFF; border: none; padding: 10px; border-radius: 5px;")
        self.output_dir_button.setFont(QFont("Segoe UI", 14))
        self.output_dir_button.clicked.connect(self.select_output_dir)

        # Start button
        self.start_button = QPushButton("Создать токенизатор")
        self.start_button.setStyleSheet("background-color: #28A745; color: #FFFFFF; border: none; padding: 10px; border-radius: 5px;")
        self.start_button.setFont(QFont("Segoe UI", 14))
        self.start_button.clicked.connect(self.create_tokenizer)

        # Layout setup
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("Токены:"), self.tokens_text_edit)
        form_layout.addRow(QLabel("Папка для сохранения:"), self.output_dir_line_edit)
        form_layout.addRow(self.output_dir_button)

        layout.addLayout(form_layout)
        layout.addWidget(self.start_button)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("background-color: #333; color: #FFFFFF; font: 12pt 'Segoe UI';")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if dir_path:
            self.output_dir_line_edit.setText(dir_path)

    def create_tokenizer(self):
        output_dir = self.output_dir_line_edit.text()
        tokens_text = self.tokens_text_edit.toPlainText()

        if not output_dir:
            self.status_bar.showMessage("Необходимо выбрать папку для сохранения.")
            return

        if not tokens_text:
            self.status_bar.showMessage("Необходимо ввести токены.")
            return

        self.status_bar.showMessage("Создание токенизатора. Пожалуйста, подождите...")

        try:
            tokens = [token.strip() for token in tokens_text.split(',') if token.strip()]
            self.generate_tokenizer(output_dir, tokens)
            self.status_bar.showMessage("Токенизатор успешно создан!")
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка: {str(e)}")

    def generate_tokenizer(self, output_dir, tokens):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        # Настройка тренера для BPE
        trainer = trainers.BpeTrainer(
            vocab_size=99,  # Настройте размер словаря
            special_tokens=tokens,
            show_progress=True
        )

        # Получение данных из текстового поля
        data = self.tokens_text_edit.toPlainText().splitlines()
        
        # Обучение токенизатора на данных
        tokenizer.train_from_iterator(data, trainer)

        # Сохранение токенизатора
        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        # Сохранение vocab.json
        vocab = tokenizer.get_vocab()
        with open(os.path.join(output_dir, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        # Сохранение merges.txt, извлекая информацию из tokenizer.json
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            merges = tokenizer_data['model']['merges']
            with open(os.path.join(output_dir, "merges.txt"), 'w', encoding='utf-8') as merges_file:
                for merge in merges:
                    merges_file.write(f"{merge}\n")

        # Сохранение special_tokens_map.json
        special_tokens = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        with open(os.path.join(output_dir, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=4)

        # Сохранение tokenizer_config.json
        with open(os.path.join(output_dir, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "max_len": 1024,  # max_len: Этот параметр указывает максимальную длину текста (в токенах), который токенизатор может обработать за один раз. Если текст превышает эту длину, он обычно будет обрезан или разделен на части. В данном случае, 1024 означает, что максимальная длина последовательности токенов для этого токенизатора составляет 1024.
                "do_lower_case": False
            }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TokenizerApp()
    window.show()
    sys.exit(app.exec())
