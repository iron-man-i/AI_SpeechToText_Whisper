import numpy as np
import ffmpeg
from pywhispercpp.model import Model
from pathlib import Path
import sys
import subprocess

def load_audio(file_path: str, sample_rate: int = 16000, ffmpeg_path: str = "ffmpeg") -> np.ndarray:
    """
    Загружает аудиофайл и конвертирует его в массив numpy с указанной частотой дискретизации.
    
    Args:
        file_path (str): Путь к аудиофайлу (WAV, MP3 и т.д.).
        sample_rate (int): Частота дискретизации (по умолчанию 16000 Гц для Whisper).
        ffmpeg_path (str): Путь к исполняемому файлу ffmpeg (по умолчанию 'ffmpeg' для PATH).
    
    Returns:
        np.ndarray: Аудиоданные в формате float32.
    
    Raises:
        FileNotFoundError: Если аудиофайл или FFmpeg не найдены.
        RuntimeError: Если FFmpeg не может обработать аудио.
    """
    try:
        # Приводим путь к Windows-формату
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {file_path}")

        # Проверяем доступность FFmpeg
        try:
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
            print(f"FFmpeg версия: {result.stdout.decode('utf-8').splitlines()[0]}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FFmpeg не найден по пути: {ffmpeg_path}. "
                "Убедитесь, что FFmpeg установлен и добавлен в PATH, или укажите правильный путь к ffmpeg.exe."
            )

        # Используем FFmpeg для конвертации аудио
        stream = ffmpeg.input(str(file_path), threads=0)
        stream = ffmpeg.output(stream, "-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
        
        # Запускаем FFmpeg
        out, err = ffmpeg.run(stream, cmd=[ffmpeg_path, "-nostdin"], capture_stdout=True, capture_stderr=True)

        if not out:
            raise RuntimeError(f"Ошибка FFmpeg: {err.decode('utf-8') if err else 'Неизвестная ошибка'}")

        # Конвертируем в numpy массив
        audio_array = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return audio_array

    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке аудио: {str(e)}")

def transcribe_audio(audio_file: str, model_name: str = "medium", ffmpeg_path: str = "ffmpeg") -> None:
    """
    Транскрибирует аудиофайл с использованием модели Whisper, выводя текст слитно.

    Args:
        audio_file (str): Путь к аудиофайлу.
        model_name (str): Название модели Whisper (например, 'base', 'medium', 'large').
        ffmpeg_path (str): Путь к исполняемому файлу ffmpeg.
    """
    try:
        # Инициализация модели с поддержкой русского языка
        print(f"Загрузка модели {model_name}...")
        model = Model(model_name, print_realtime=False, print_progress=True, language="ru")

        # Загрузка аудио
        print(f"Чтение аудиофайла {audio_file}...")
        audio_data = load_audio(audio_file, ffmpeg_path=ffmpeg_path)

        # Транскрипция
        print("Начало транскрипции...")
        segments = model.transcribe(audio_data)
        
        # Объединяем текст из всех сегментов
        full_text = " ".join(segment.text.strip() for segment in segments)
        
        # Вывод и сохранение результатов
        print("\nРезультат транскрипции:")
        print(full_text)
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

    except Exception as e:
        print(f"Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Путь к вашему аудиофайлу с русской лекцией
    audio_file_path = r"C:\Users\Iron_Man\Desktop\1.mp3"  # Замените на актуальный путь
    # Путь к ffmpeg.exe

    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Убедитесь, что путь правильный
    transcribe_audio(audio_file_path, model_name="medium", ffmpeg_path=ffmpeg_path)