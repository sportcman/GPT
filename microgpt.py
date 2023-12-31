from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Импорт необходимых классов из библиотеки Transformers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Создаем конфигурацию модели с большими параметрами
model_config = GPT2Config.from_pretrained('ai-forever/rugpt2large')  # Загрузка предварительно обученной конфигурации модели GPT-2.
model_config.n_layer = 120  # Установка числа слоев модели, это позволит модели обнаруживать более сложные зависимости в данных.
model_config.n_head = 10  # Установка числа attention-головок модели, это помогает модели обрабатывать различные аспекты контекста одновременно.
model_config.n_embd = 160  # Установка размерности векторного представления (embedding), более высокая размерность позволяет модели учиться более сложным взаимосвязям между словами. 
model_config.intermediate_size = 640  # Установка размера промежуточного слоя модели, более высокое значение может помочь модели обрабатывать более сложные взаимосвязи между словами.
model_config.hidden_size = 960  # Установка размера скрытого состояния модели,чтобы увеличить количество параметров и, следовательно, емкость модели, большая емкость может помочь модели представлять более сложные зависимости в данных.
model_config.mem_len = 640  # Установка длины оперативной памяти модели, Более длинный контекст может повысить качество и разнообразие сгенерированного текста.

# Создаем модель на основе заданной конфигурации
model = GPT2LMHeadModel(config=model_config)  # Создание объекта модели GPT-2 с головой для языкового моделирования на основе заданной конфигурации

# Создаем токенизатор
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/rugpt2large')  # Загрузка предварительно обученного токенизатора GPT-2

# Сохраняем модель и токенизатор
model.save_pretrained('C:/Novak/model_complex')  # Сохранение модели в указанной директории
tokenizer.save_pretrained('C:/Novak/model_complex')  # Сохранение токенизатора в указанной директории
