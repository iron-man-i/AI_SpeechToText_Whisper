import whisper # pip install git+https://github.com/openai/whisper.git
import os # pip install os

LECTURES_DIR = "/root/PROJECT/SOURCE"   # Путь к папке с аудиофайлами
TEXTS_DIR = "/root/PROJECT/RESULT"      # Путь к папке для сохранения текстовых файлов
model = whisper.load_model("large") # Можно выбрать тип модели large, medium,base, tiny, small

for filename in os.listdir(LECTURES_DIR): # Перебираем все файлы в папке LECTURES_DIR
    if filename.lower().endswith('.mp3'):  # Можно выбрать формат mp3, wav, m4a, flac, etc.
        audio_path = os.path.join(LECTURES_DIR, filename) # Путь к аудиофайлу
        text_path = os.path.join(TEXTS_DIR, os.path.splitext(filename)[0] + ".txt")  # Путь к текстовому файлу
        try: # Обработка ошибок
            print(f"Транскрибирую {audio_path} ...") # Выводим сообщение о том, что транскрибируем аудиофайл
            result = model.transcribe(audio_path, language='ru') # language='ru' - для русского языка, language='en' - для английского языка
            with open(text_path, "w", encoding="utf-8") as f: # Открываем текстовый файл для записи
                f.write(result['text']) # Записываем результат в текстовый файл
            print(f"Текст сохранён в {text_path}") # Выводим сообщение о том, что текст сохранён
        except Exception as e: # Обработка ошибок
            print(f"Ошибка при обработке {audio_path}: {e}") # Выводим сообщение о том, что произошла ошибка