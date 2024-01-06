import re

def save_to_file(sentences, output_filename):
    with open(output_filename, "w", encoding="utf-8") as output_file:
        for sentence in sentences:
            if sentence.strip():
                output_file.write(sentence.strip() + '\n')

def main(input_filename, output_filename):
    with open(input_filename, "r", encoding="utf-8") as input_file:
        text = input_file.read()

    # Заменяем переносы строк на пробелы, чтобы не разбивать предложения, которые переносятся на следующую строку
    text = text.replace("\n", " ")

    # Разделение текста на предложения по знакам пунктуации (.?!)
    sentences = re.split(r"(?<=[.?!])\s+", text)

    # Удаляем пустые строки
    sentences = [sentence for sentence in sentences if sentence.strip()]

    # Записываем предложения в выходной файл
    save_to_file(sentences, output_filename)

if __name__ == "__main__":
    input_filename = "C:\\txt\\input.txt"  # Имя входного файла
    output_filename = "C:\\txt\\output.txt"  # Имя выходного файла

    main(input_filename, output_filename)
