# Решение для работы на GPU

Здесь представлено решение для работы на GPU (предпочтительно на сервере).


Для установки библиотек в терминале нужно ввести следующие команды:


sudo apt update && sudo apt install -y ffmpeg

pip install -U openai-whisper

python 3 (путь к файлу Whisper_GPU.py для запуска)


Примечания: 

0) Решение предназначено преимущественно для распознавания текстов из аудиолекций курсом(-ами).
1) Лучше всего распознаёт текст из аудио модель "large" (объем 2,88 GB).
2) Работает нейронная сеть "large" долго: в среднем 1 минута аудио в .mp3 распознаётся за 30 секунд, но результат стоит того, чтобы его ожидать.
3) Для оптимальной работы этой модели нужна видеокарта 16 GB (например Tesla T4). Увеличение мощности видеокарты (например до 24 GB) слабо коррелирует со скоростью работы модели.
4) Решение хорошо работает в бесплатной версии Google Colab (хватает бесплатного ресурса Google Colab, чтобы распознать от 2 до 4-х лекций продолжительностью от 1 до 1,5 часов).
5) Модель с дополнительными настройками может сразу переводить распознаваемый текст на другой язык.
