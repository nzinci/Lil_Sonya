# Sonia

Для генерации музыки потребуются следующие библиотеки для установки:

    torch       pretty_midi         fluidsynth
    numpy       progress.bar
    optprase    midi2audio

Запускается со следующими параметрами:

    -с      control         Это либо руками написанные распределения, либо путь к файлу 
                                в котором уже заданы они (лучший вариант)
    -b      batch_size      По дефолту стоит 6
    -s      session path    Путь к файлу с моделью  (называется она final_2.sess)
    -o      output dir      Куда генерим сэмплы
    -l      length          Длина сэмлпа (лучше брать 1100)
    -f      font_path       Путь к файлу с soundfont (по дефолту он там же, где и исполняемый файл)

Пример такого запуска:

    python3 gen_f.py -s final_2.sess -l 1000 -c ./test_train/beethoven.data -o ./gen_mus -b 8


Для воспроизведения музыки настоятельно рекомендуется плеер timidity++, так как 
при конвертации midi в wav очень влияет этот soundfont.
По ссылке можно найти на диске датасет, обработанный датасет, модель (final_2.sess возможно лучше)
и soundfont.
https://drive.google.com/open?id=1TwJhkTkN6UmYd3v9ky0Dak8fqpbHyaXk
