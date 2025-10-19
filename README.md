# Speech Enhancement (MP-SENet + Mamba)

## Аннотация

Реализована модифицированная версия [MP-SENet](https://github.com/yxlu-0102/MP-SENet) для улучшения речи на **VoiceBank+DEMAND (16 kHz)** [1]. Классические Transformer-блоки заменены на параллельное сочетание Transformer и [Mamba (mamba-ssm)](https://github.com/state-spaces/mamba) [2]; обе ветви обрабатывают признаки независимо, их выходы объединяются через **cross-attention**. Модель достигла **PESQ=3.44** на тестовом наборе. Репозиторий содержит код модели, интерактивное демо (Gradio) и ноутбук с полным воспроизводимым пайплайном: подготовка окружения и данных, обучение, подсчёт метрик и краткий анализ.

---

## Структура репозитория

```
.
├─ MP-SENet/                 # код модели, основан на github.com/yxlu-0102/MP-SENet с изменением архитектуры
│  ├─ models/
│  │  └─ model.py            # реализация варианта MP-SENet с параллельными Transformer и Mamba + cross-attention
│  └─ ...                    # прочие модули обучения/инференса
├─ gradio_app/               # код демо-приложения (см. отдельный README внутри папки)
│  └─ README.md
└─ model_showcase.ipynb      # Основной файл: там настраивается окружение, обучается модель (можно взять готовую), считаются и анализируются метрики
```

---

## Модель

**Идея.** В базовой MP-SENet временно-частотные признаки проходят через Transformer-блоки. В данной работе они заменены на **двухуровневую схему**:

* **Transformer-ветвь** хорошо улавливает главные зависимости, как в исходном коде;
* **Mamba-ветвь** — позволяет углубить модель, не сильно влияя на latency (потому что ssm модель).
  Выходы ветвей объединяются **cross-attention’ом**, затем подаются в параллельные декодеры **Magnitude / Phase** (как в MP-SENet). Подробности см. в `MP-SENet/models/model.py`.

Общее число параметров составило 5.95М, но так как основной прирост числа параметров получился из-за добавления мамбы, то общий latency не сильно увеличился (из-за оптимизаций на CUDA и линейного внимания).

---

## Как пользоваться

1. **Jupyter Notebook:** откройте `model_showcase.ipynb` и следуйте разделам по порядку
   (установка зависимостей, загрузка VoiceBank+DEMAND, обучение/загрузка готовой модели, подсчёт метрик, визуализации).
2. **Демо-приложение на Gradio:** перейдите в `gradio_app/` и используйте инструкции из локального `README.md`.

---

## Результаты

На VoiceBank+DEMAND:

| Метрика                      |               Значение |
| ---------------------------- | ---------------------: |
| **PESQ (wb)**                |               **3.44** |
| **CSIG**                     |               **4.73** |
| **CBAK**                     |               **3.93** |
| **COVL**                     |               **4.20** |
| **STOI**                     |               **0.96** |
| **SSNR (dB)**                |              **10.67** |
| **UTMOS**                    |               **4.05** |
| **DNSMOS SIG / BAK / P.808** | **3.53 / 4.07 / 3.56** |
| **NISQA MOS**                |               **4.60** |
| **SCOREQ**                   |               **4.40** |

---

## Репликация эксперимента

Все шаги подробно описаны в `model_showcase.ipynb`:

* подготовка окружения и зависимостей;
* скачивание и подготовка VoiceBank+DEMAND;
* обучение/валидация модифицированной MP-SENet;
* вычисление non-intrusive и intrusive-метрик;
* сравнение noisy / enhanced / clean, спектрограммы и аудио-примеры (аудио не ставится, если смотреть через github).

---

## Примечания

* Обучение модели с нуля до PESQ > 3.2 занимает 8-10 часов на ноутбучной 3070 ti
* Несмотря на то, что mamba позволяет сильно ускорить вычисления как на инференсе, так и в обучении, важно помнить, что для её использования необходим CUDA, что делает модель непригодной для edge инференса.

---
## Источники

1. **Lu et al.**, *MP-SENet: Magnitude-Phase Speech Enhancement Network*, *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2023.
   [https://github.com/yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet)

2. **Gu et al.**, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, *arXiv:2312.00752*, 2023.
   [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

3. **Valentini-Botinhao et al.**, *Speech Enhancement Challenge Dataset (VoiceBank+DEMAND)*, *IEEE ICASSP*, 2016.
   [https://datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)

4. **Lo et al.**, *Non-Intrusive Speech Quality Assessment Using Neural Networks*, *IEEE Signal Processing Letters*, 2019 (DNSMOS).
   [https://github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)

5. **Mittag & Möller**, *NISQA: A Deep Neural Network for Non-Intrusive Speech Quality Assessment*, *Interspeech 2021*.
   [https://github.com/gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA)

6. **Hu et al.**, *Perceptual Objective Listening Quality Assessment (POLQA)*, *ITU-T P.863*, 2018.
   (для PESQ/POLQA как бенчмарков)
