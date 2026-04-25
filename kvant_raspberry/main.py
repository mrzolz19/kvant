import speech_recognition as sr
#import re
import random
import pyttsx3
import sys
import time
import argparse
from contextlib import suppress
from configparser import ConfigParser
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' #убираем вывод от pygame
from pygame import mixer
import requests
import uuid
from openwakeword.model import Model
import pyaudio
import numpy as np
#import keyboard

try:
    import msvcrt  # только для Windows: чтение нажатий клавиш из консоли
except ImportError:
    msvcrt = None

try:
    import select
    import termios
    import tty
except ImportError:
    select = None
    termios = None
    tty = None

skip_tts_key = "q"


"""class VoiceAssistant:
    def __init__(self):
        self.speech_input = SpeechInput()
        self.speech_output = SpeechOutput()
        self.command_processor = ComandProcessor()
        self.config = Config()

class SpeechInput:
    pass

class ComandProcessor:
    pass

class SpeechOutput:
    pass

class Config:
    pass"""

class MicrophoneManager:
    def __init__(self):
        self.mic = sr.Microphone()
        self.is_active = False
        self.audio_stream = None

    def __enter__(self):
        if not self.is_active:
            self.audio_stream = self.mic.__enter__()
            self.is_active = True
        return self.audio_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_active:
            self.mic.__exit__(exc_type, exc_val, exc_tb)
            self.is_active = False
            self.audio_stream = None

    def control(self, enable: bool) -> None:  # управление микрофоном (вкл, выкл)
        if enable and not self.is_active:
            self.__enter__()
        elif not enable and self.is_active:
            self.__exit__(None, None, None)


class KeyboardSkipController:
    def __init__(self, skip_key: str):
        self.skip_key = skip_key.lower()
        self.is_windows = sys.platform.startswith("win")
        self.enabled = False
        self._stdin_fd = None
        self._stdin_state = None

    def start(self) -> None:
        if self.is_windows:
            self.enabled = bool(msvcrt)
            if self.enabled:
                self._drain_buffer()
            return

        if not (select and termios and tty):
            self.enabled = False
            return

        tcgetattr = getattr(termios, "tcgetattr", None)
        setcbreak = getattr(tty, "setcbreak", None)
        if not (tcgetattr and setcbreak):
            self.enabled = False
            return

        if not sys.stdin.isatty():
            self.enabled = False
            return

        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_state = tcgetattr(self._stdin_fd)
            setcbreak(self._stdin_fd)
            self.enabled = True
            self._drain_buffer()
        except Exception:
            self.enabled = False
            self._stdin_fd = None
            self._stdin_state = None

    def stop(self) -> None:
        if self.is_windows:
            return

        if not (self.enabled and self._stdin_fd is not None and self._stdin_state is not None):
            return

        tcsetattr = getattr(termios, "tcsetattr", None)
        tcsadrain = getattr(termios, "TCSADRAIN", None)
        if not (tcsetattr and tcsadrain is not None):
            return

        with suppress(Exception):
            tcsetattr(self._stdin_fd, tcsadrain, self._stdin_state)

    def _drain_buffer(self) -> None:
        if not self.enabled:
            return

        if self.is_windows:
            while msvcrt and msvcrt.kbhit():
                msvcrt.getwch()
            return

        while self._stdin_has_data():
            with suppress(Exception):
                sys.stdin.read(1)

    def _stdin_has_data(self) -> bool:
        if not (select and self._stdin_fd is not None):
            return False

        select_fn = getattr(select, "select", None)
        if not select_fn:
            return False

        readable, _, _ = select_fn([self._stdin_fd], [], [], 0)
        return bool(readable)

    def is_skip_pressed(self) -> bool:
        if not self.enabled:
            return False

        if self.is_windows:
            while msvcrt and msvcrt.kbhit():
                if msvcrt.getwch().lower() == self.skip_key:
                    return True
            return False

        while self._stdin_has_data():
            try:
                if sys.stdin.read(1).lower() == self.skip_key:
                    return True
            except Exception:
                return False
        return False

def text_playback(text: str) -> None: #озвучивние текста
    mic_manager.control(False)
    mixer.quit()  # Освобождаем аудиоустройство от pygame, иначе pyttsx3 обрезает речь
    text = text.replace('*', '').replace('`', '').replace("#", "")
    #text = re.sub(r'[^\w\s,.!?;:\'"\-]', '', text)
    engine = pyttsx3.init()  # Пересоздаём engine для корректной работы runAndWait
    skip_controller = KeyboardSkipController(skip_tts_key)
    loop_started = False

    skip_controller.start()
    try:
        if skip_controller.enabled:
            print(f"Нажмите '{skip_tts_key}', чтобы пропустить озвучку")

        engine.say(text)
        engine.startLoop(False)
        loop_started = True

        while engine.isBusy():
            engine.iterate()
            if skip_controller.is_skip_pressed():
                print("Озвучка прервана")
                engine.stop()
                break
            time.sleep(0.01)
    finally:
        if loop_started:
            with suppress(Exception):
                engine.endLoop()
        skip_controller.stop()
        with suppress(Exception):
            engine.stop()
        time.sleep(0.3)  # Даём время на завершение воспроизведения аудиобуфера
        mixer.init()  # Повторная инициализация pygame mixer

def voicing_greetings(): #функция приветствия после активационное фразы
    mixer.music.load(f"sound/greet{random.choice([1, 2, 3])}.wav")
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.05)  # Ждём завершения воспроизведения без нагрузки CPU
    mic_manager.control(False)
    print("К вашим услугам, сэр")

def request_processing(text: str) -> str: #функция ответа нейросетью
    data = {
        "chatInput": text,
        "sessionId": session_id
    }
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(webhook_n8n, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            print(result['output'])
            return result['output']
        except requests.exceptions.HTTPError as e:
            print(f"Ошибка HTTP (попытка {attempt+1}/{max_retries+1}): {e}")
            # Для 502/503 пробуем повторить
            if attempt < max_retries and response.status_code in (502, 503, 504):
                import time
                time.sleep(2 * (attempt + 1))
                continue
            return "Извините, сервер временно недоступен."
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к n8n: {e}")
            return "Извините, произошла ошибка при обработке запроса."


def command_processing():
    try:
        while True:
            with mic_manager:
                try:
                    print("Слушаю...")
                    recognizer.adjust_for_ambient_noise(source=mic_manager.mic, duration=0.65)
                    audio = recognizer.listen(mic_manager.mic, timeout=timeout)
                    text = recognizer.recognize_google(audio, language="ru")
                    text_for_cmd = text.lower().strip().replace('!', '').replace('.', '').replace('?', '').replace(',', '')
                    print(f"Вы сказали: {text}")
                    handled = False


                    #Команды:
                    if text_for_cmd in cmd_exit:
                        print("Отключаю питание")
                        mixer.music.load("sound/off_power.wav")
                        mixer.music.play()
                        sys.exit()
                        handled = True

                    if not text.strip():
                        print("Пустая команда")
                        continue

                #Анализ сказанного:
                except sr.WaitTimeoutError:
                    print("Вы где? Жду команды...")
                    return
                except sr.UnknownValueError:
                    print("Речь не распознана")
                    continue

                if not handled:
                    text_playback(request_processing(text))
                    break

    except sr.RequestError as e:
        print(f"Ошибка сервиса: {e}")
    except Exception as e:
        print(f"Ошибка: {e}")

def main():
    mixer.init()
    mixer.music.load("sound/run.wav")
    mixer.music.play()
    try:
        oww_model = Model(
            wakeword_models=[model_path],
            inference_framework='onnx'
        )
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # Параметры аудиопотока
    sample_rate = 16000  # частота дискретизации, поддерживаемая моделью
    chunk_size = 1280    # размер блока для обработки моделью

    # Инициализация PyAudio
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    print("Ожидаю активационную фразу...")
    while True:
        # Чтение аудиоданных напрямую из потока
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)

        # Предсказание wake word
        prediction = oww_model.predict(audio_data)
        if prediction['Quant'] > 0.25:
            stream.stop_stream()
            oww_model.reset()
            voicing_greetings()
            command_processing()
            stream.start_stream()
            print("Ожидаю активационную фразу...")

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Jarvis Voice Assistant')
    parser.add_argument('--webhook', type=str, help='URL webhook для n8n')
    args = parser.parse_args()

    mic_manager = MicrophoneManager()
    model_path = "Quant.onnx"
    config = ConfigParser()
    with open("settings.ini", "r", encoding="utf-8") as f:
        config.read_file(f)

    cmd_exit = config["Commands"]["Cmd_Exit"]
    #Speech Recognition
    timeout = int(config["Speech"]["TimeoutSpeechRecognition"]) #через сколько секунд снова обращаться к wake word после отсуствия звуков

    # Распознователь речи speech_recognition
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1 #фраза будет завершённой после этого таймаута в сек

    # pyttsx3 engine создаётся в text_playback() при каждом вызове,
    # чтобы избежать бага с обрезкой речи при повторных runAndWait()

    # Проверяем webhook n8n (приоритет: аргумент командной строки -> config -> ввод пользователя)
    if args.webhook:
        webhook_n8n = args.webhook
        print(f"Используется webhook из аргумента: {webhook_n8n}")
        # Сохраняем в конфиг для последующих запусков
        config['Settings']['webhook_n8n'] = webhook_n8n
        with open('settings.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)
    elif not config['Settings']['webhook_n8n']:
        webhook_n8n = input("Введите webhook n8n: ")
        config['Settings']['webhook_n8n'] = webhook_n8n
        # Записываем изменения в файл
        with open('settings.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)
    else:
        webhook_n8n = config['Settings']['webhook_n8n']

    # Модели OpenWakeWord уже предзагружены в Docker-образе на этапе сборки
    # Дополнительная загрузка не требуется
    session_id = str(uuid.uuid4())
    main()
