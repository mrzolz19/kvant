import argparse
import ast
import io
import random
import re
import sys
import time
import wave
import uuid
from collections import deque
from configparser import ConfigParser
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from os import environ
from pathlib import Path

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from dotenv import load_dotenv
from groq import Groq
import numpy as np
import pyaudio
import pyttsx3
import requests
import webrtcvad
from openwakeword.model import Model
from pygame import mixer

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


CONFIG_PATH = Path("settings.ini")
MODEL_PATH = Path("Quant.onnx")
SOUND_DIR = Path("sound")
SKIP_TTS_KEY = "q"
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
WAKEWORD_NAME = "Quant"
UNINTELLIGIBLE_VOICE_MESSAGE = (
    "Голосовое сообщение не содержит разборчивой речи. "
    "Пожалуйста, повторите команду четче."
)
NO_SPEECH_PROB_THRESHOLD = 0.6
LOW_CONFIDENCE_AVG_LOGPROB_THRESHOLD = -1.0
NO_SPEECH_WITH_LOW_CONFIDENCE_THRESHOLD = 0.45
NO_SPEECH_SEGMENT_RATIO_THRESHOLD = 0.8

load_dotenv()


@dataclass(frozen=True)
class SpeechConfig:
    timeout: float
    language: str
    min_text_chars: int
    groq_api_key: str
    groq_model: str
    groq_temperature: float


@dataclass(frozen=True)
class VadConfig:
    aggressiveness: int
    frame_ms: int
    pre_roll_ms: int
    start_speech_ms: int
    min_speech_ms: int
    end_silence_ms: int
    command_timeout: float
    tail_padding_ms: int


@dataclass(frozen=True)
class AppConfig:
    webhook_n8n: str
    cmd_exit: set[str]
    speech: SpeechConfig
    vad: VadConfig
    request_timeout: float
    request_retries: int


class ConfigLoader:
    def __init__(self, path: Path):
        self.path = path
        self.parser = ConfigParser()

    def load(self, webhook_override: str | None = None) -> AppConfig:
        with self.path.open("r", encoding="utf-8") as file:
            self.parser.read_file(file)

        webhook = self._resolve_webhook(webhook_override)
        return AppConfig(
            webhook_n8n=webhook,
            cmd_exit=self._commands(),
            speech=SpeechConfig(
                timeout=self.parser.getfloat(
                    "Speech", "TimeoutSpeechRecognition", fallback=8.0
                ),
                language=self.parser.get("Speech", "Language", fallback="ru-RU"),
                min_text_chars=self.parser.getint("Speech", "MinTextChars", fallback=3),
                groq_api_key=self._groq_api_key(),
                groq_model=self.parser.get(
                    "Speech", "GroqModel", fallback="whisper-large-v3"
                ),
                groq_temperature=self.parser.getfloat(
                    "Speech", "GroqTemperature", fallback=0.0
                ),
            ),
            vad=VadConfig(
                aggressiveness=self.parser.getint("Vad", "Aggressiveness", fallback=2),
                frame_ms=self.parser.getint("Vad", "FrameMs", fallback=30),
                pre_roll_ms=self.parser.getint("Vad", "PreRollMs", fallback=300),
                start_speech_ms=self.parser.getint(
                    "Vad", "StartSpeechMs", fallback=300
                ),
                min_speech_ms=self.parser.getint("Vad", "MinSpeechMs", fallback=450),
                end_silence_ms=self.parser.getint("Vad", "EndSilenceMs", fallback=900),
                command_timeout=self.parser.getfloat(
                    "Vad", "CommandTimeout", fallback=15.0
                ),
                tail_padding_ms=self.parser.getint(
                    "Vad", "TailPaddingMs", fallback=250
                ),
            ),
            request_timeout=self.parser.getfloat(
                "Settings", "RequestTimeout", fallback=180.0
            ),
            request_retries=self.parser.getint(
                "Settings", "RequestRetries", fallback=2
            ),
        )

    def _resolve_webhook(self, webhook_override: str | None) -> str:
        if webhook_override:
            self.parser["Settings"]["webhook_n8n"] = webhook_override
            self._save()
            print(f"Используется webhook из аргумента: {webhook_override}")
            return webhook_override

        webhook = self.parser.get("Settings", "webhook_n8n", fallback="").strip()
        if webhook:
            return webhook

        webhook = input("Введите webhook n8n: ").strip()
        self.parser["Settings"]["webhook_n8n"] = webhook
        self._save()
        return webhook

    def _commands(self) -> set[str]:
        raw_commands = self.parser.get("Commands", "Cmd_Exit", fallback="[]")
        try:
            commands = ast.literal_eval(raw_commands)
        except (SyntaxError, ValueError):
            commands = []

        if not isinstance(commands, (list, tuple, set)):
            commands = []

        return {
            normalize_command(str(command))
            for command in commands
            if str(command).strip()
        }

    def _groq_api_key(self) -> str:
        from os import getenv

        return (
            getenv("GROQ_API_KEY", "").strip()
            or self.parser.get("Speech", "GroqApiKey", fallback="").strip()
        )

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            self.parser.write(file)


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
        if not (tcgetattr and setcbreak) or not sys.stdin.isatty():
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

        if not (
            self.enabled
            and self._stdin_fd is not None
            and self._stdin_state is not None
        ):
            return

        tcsetattr = getattr(termios, "tcsetattr", None)
        tcsadrain = getattr(termios, "TCSADRAIN", None)
        if tcsetattr and tcsadrain is not None:
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


class SoundPlayer:
    def init(self) -> None:
        if not mixer.get_init():
            mixer.init()

    def quit(self) -> None:
        with suppress(Exception):
            mixer.quit()

    def play(self, path: Path) -> None:
        self.init()
        mixer.music.load(str(path))
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.05)


class SpeechOutput:
    def __init__(self, sound_player: SoundPlayer, skip_key: str = SKIP_TTS_KEY):
        self.sound_player = sound_player
        self.skip_key = skip_key

    def speak(self, text: str) -> None:
        text = clean_tts_text(text)
        if not text:
            return

        self.sound_player.quit()
        engine = pyttsx3.init()
        skip_controller = KeyboardSkipController(self.skip_key)
        loop_started = False

        skip_controller.start()
        try:
            if skip_controller.enabled:
                print(f"Нажмите '{self.skip_key}', чтобы пропустить озвучку")

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
            time.sleep(0.3)
            self.sound_player.init()


class WakeWordListener:
    def __init__(self, audio_interface: pyaudio.PyAudio, model_path: Path):
        self.audio_interface = audio_interface
        self.model = Model(
            wakeword_models=[str(model_path)], inference_framework="onnx"
        )
        self.chunk_size = 1280
        self.stream = None

    def start(self) -> None:
        if self.stream is not None:
            return

        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    def wait(self) -> None:
        self.start()
        print("Ожидаю активационную фразу...")
        while True:
            audio_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            prediction = self.model.predict(audio_data)

            if prediction.get(WAKEWORD_NAME, 0) > 0.25:
                self.model.reset()
                self.close_stream()
                return

    def close_stream(self) -> None:
        if self.stream is None:
            return

        with suppress(Exception):
            if self.stream.is_active():
                self.stream.stop_stream()
        with suppress(Exception):
            self.stream.close()
        self.stream = None

    def close(self) -> None:
        self.close_stream()


class VadCommandRecorder:
    def __init__(self, audio_interface: pyaudio.PyAudio, config: VadConfig):
        self.audio_interface = audio_interface
        self.config = self._validate_config(config)
        self.vad = webrtcvad.Vad(self.config.aggressiveness)
        self.frame_samples = int(SAMPLE_RATE * self.config.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * SAMPLE_WIDTH_BYTES
        self.pre_roll_frames = ms_to_frames(
            self.config.pre_roll_ms, self.config.frame_ms
        )
        self.start_speech_frames = ms_to_frames(
            self.config.start_speech_ms, self.config.frame_ms
        )
        self.min_speech_frames = ms_to_frames(
            self.config.min_speech_ms, self.config.frame_ms
        )
        self.end_silence_frames = ms_to_frames(
            self.config.end_silence_ms, self.config.frame_ms
        )
        self.tail_padding_bytes = int(
            SAMPLE_RATE * SAMPLE_WIDTH_BYTES * self.config.tail_padding_ms / 1000
        )
        self.sound_player = SoundPlayer()

    def record(self, start_timeout: float) -> bytes | None:
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.frame_samples,
        )
        print("Слушаю...")

        try:
            frames, speech_frame_count = self._record_frames(stream, start_timeout)
        finally:
            with suppress(Exception):
                stream.stop_stream()
            with suppress(Exception):
                stream.close()

        if not frames:
            return None
        if speech_frame_count < self.min_speech_frames:
            speech_ms = speech_frame_count * self.config.frame_ms
            print(f"Слишком короткая команда ({speech_ms} мс речи), игнорирую")
            return None

        return b"".join(frames) + (b"\x00" * self.tail_padding_bytes)

    def _record_frames(self, stream, start_timeout: float) -> tuple[list[bytes], int]:
        pre_roll = deque(maxlen=self.pre_roll_frames)
        recorded: list[bytes] = []
        speech_frames = 0
        recorded_speech_frames = 0
        silence_frames = 0
        recording = False
        wait_started_at = time.monotonic()
        record_started_at = None

        while True:
            frame = stream.read(self.frame_samples, exception_on_overflow=False)
            if len(frame) != self.frame_bytes:
                continue

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if not recording:
                pre_roll.append(frame)
                speech_frames = speech_frames + 1 if is_speech else 0

                if speech_frames >= self.start_speech_frames:
                    recording = True
                    record_started_at = time.monotonic()
                    recorded.extend(pre_roll)
                    recorded_speech_frames = speech_frames
                    silence_frames = 0
                    print("Речь обнаружена, записываю команду...")
                    continue

                if time.monotonic() - wait_started_at >= start_timeout:
                    print("Вы где? Жду команды...")
                    return [], 0
                continue

            recorded.append(frame)
            if is_speech:
                recorded_speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            if silence_frames >= self.end_silence_frames:
                request_complished_sir = SOUND_DIR / "request.wav"
                self.sound_player.play(request_complished_sir)
                print("Команда записана")
                return recorded, recorded_speech_frames

            if (
                record_started_at
                and time.monotonic() - record_started_at >= self.config.command_timeout
            ):
                print("Достигнут лимит длительности команды")
                return recorded, recorded_speech_frames

    def _validate_config(self, config: VadConfig) -> VadConfig:
        if config.frame_ms not in (10, 20, 30):
            raise ValueError("Vad.FrameMs должен быть 10, 20 или 30")
        if not 0 <= config.aggressiveness <= 3:
            raise ValueError("Vad.Aggressiveness должен быть от 0 до 3")
        return config


class SpeechRecognizer:
    def __init__(
        self,
        language: str,
        min_text_chars: int,
        groq_api_key: str,
        groq_model: str,
        groq_temperature: float,
    ):
        self.language = language
        self.min_text_chars = min_text_chars
        self.groq_model = groq_model
        self.groq_temperature = groq_temperature
        self.client = Groq(api_key=groq_api_key or None)

    def recognize(self, audio_bytes: bytes) -> str | None:
        try:
            transcription = self._transcribe(audio_bytes)
        except Exception as error:
            print(f"Ошибка Groq Speech-to-Text: {error}")
            return None

        if is_unintelligible_transcription(transcription):
            print(UNINTELLIGIBLE_VOICE_MESSAGE)
            return None

        text = str(metadata_value(transcription, "text", "") or "").strip()
        if not text:
            print("Пустая команда")
            return None
        if len(normalize_command(text).replace(" ", "")) < self.min_text_chars:
            print(f"Слишком короткий распознанный текст: {text!r}")
            return None

        print(f"Вы сказали: {text}")
        return text

    def _transcribe(self, audio_bytes: bytes) -> Any:
        wav_bytes = pcm16_to_wav_bytes(audio_bytes, SAMPLE_RATE, SAMPLE_WIDTH_BYTES)
        return self.client.audio.transcriptions.create(
            file=("command.wav", wav_bytes),
            model=self.groq_model,
            language=groq_language(self.language),
            temperature=self.groq_temperature,
            response_format="verbose_json",
        )


class N8nClient:
    def __init__(self, webhook_url: str, session_id: str, timeout: float, retries: int):
        self.webhook_url = webhook_url
        self.session_id = session_id
        self.timeout = timeout
        self.retries = retries

    def ask(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Я не расслышал вопрос. Повторите, пожалуйста."

        payload = {"chatInput": text, "sessionId": self.session_id}
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.webhook_url, json=payload, timeout=self.timeout
                )
                if response.status_code in (502, 503, 504) and attempt < self.retries:
                    self._retry_pause(attempt)
                    continue

                response.raise_for_status()
                result = response.json()
                answer = self._extract_output(result)
                if not isinstance(answer, str) or not answer.strip():
                    print(f"Некорректный ответ n8n: {result}")
                    return "Извините, сервер вернул пустой ответ."

                answer = answer.strip()
                print(answer)
                return answer
            except requests.exceptions.HTTPError as error:
                print(
                    f"Ошибка HTTP (попытка {attempt + 1}/{self.retries + 1}): {error}"
                )
                return "Извините, сервер временно недоступен."
            except requests.exceptions.RequestException as error:
                print(f"Ошибка запроса к n8n: {error}")
                if attempt < self.retries:
                    self._retry_pause(attempt)
                    continue
                return "Извините, произошла ошибка при обработке запроса."
            except ValueError as error:
                print(f"n8n вернул не JSON: {error}")
                return "Извините, сервер вернул некорректный ответ."

        return "Извините, сервер временно недоступен."

    def _retry_pause(self, attempt: int) -> None:
        time.sleep(2 * (attempt + 1))

    def _extract_output(self, result):
        if isinstance(result, dict):
            return result.get("output")
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0].get("output")
        return None


class VoiceAssistant:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.audio_interface = pyaudio.PyAudio()
        self.sound_player = SoundPlayer()
        self.speech_output = SpeechOutput(self.sound_player)
        self.speech_recognizer = SpeechRecognizer(
            language=config.speech.language,
            min_text_chars=config.speech.min_text_chars,
            groq_api_key=config.speech.groq_api_key,
            groq_model=config.speech.groq_model,
            groq_temperature=config.speech.groq_temperature,
        )
        self.n8n_client = N8nClient(
            webhook_url=config.webhook_n8n,
            session_id=self.session_id,
            timeout=config.request_timeout,
            retries=config.request_retries,
        )
        self.wake_listener = WakeWordListener(self.audio_interface, MODEL_PATH)
        self.command_recorder = VadCommandRecorder(self.audio_interface, config.vad)

    def run(self) -> None:
        try:
            self.sound_player.init()
            self.sound_player.play(SOUND_DIR / "with_reference_cer.wav")

            while True:
                self.wake_listener.wait()
                self._handle_wakeword()
        except KeyboardInterrupt:
            print("\nЗавершение работы")
        finally:
            self.close()

    def _handle_wakeword(self) -> None:
        self._play_greeting()
        audio = self.command_recorder.record(start_timeout=self.config.speech.timeout)
        if audio is None:
            return

        text = self.speech_recognizer.recognize(audio)
        if text is None:
            return

        if normalize_command(text) in self.config.cmd_exit:
            self._shutdown()

        answer = self.n8n_client.ask(text)
        self.speech_output.speak(answer)

    def _play_greeting(self) -> None:
        greeting_path = SOUND_DIR / f"greet{random.choice([1, 2, 3])}.wav"
        self.sound_player.play(greeting_path)
        print("К вашим услугам, сэр")

    def _shutdown(self) -> None:
        print("Отключаю питание")
        self.sound_player.play(SOUND_DIR / "off_power.wav")
        raise SystemExit(0)

    def close(self) -> None:
        self.wake_listener.close()
        with suppress(Exception):
            self.audio_interface.terminate()
        self.sound_player.quit()


def ms_to_frames(duration_ms: int, frame_ms: int) -> int:
    return max(1, int(round(duration_ms / frame_ms)))


def normalize_command(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text)


def clean_tts_text(text: str) -> str:
    return text.replace("*", "").replace("`", "").replace("#", "").strip()


def pcm16_to_wav_bytes(audio_bytes: bytes, sample_rate: int, sample_width: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


def groq_language(language: str) -> str:
    return language.split("-", maxsplit=1)[0].lower() if language else "ru"


def metadata_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def is_unintelligible_transcription(transcription: Any) -> bool:
    recognized_text = (metadata_value(transcription, "text", "") or "").strip()
    if not recognized_text:
        return True

    segments = metadata_value(transcription, "segments", None) or []
    if not segments:
        return False

    no_speech_probs = _collect_float_metadata(segments, "no_speech_prob")
    avg_logprobs = _collect_float_metadata(segments, "avg_logprob")
    if not no_speech_probs and not avg_logprobs:
        return False

    mean_no_speech_prob = _mean(no_speech_probs)
    mean_avg_logprob = _mean(avg_logprobs)
    high_no_speech_ratio = _ratio_at_least(
        no_speech_probs,
        NO_SPEECH_PROB_THRESHOLD,
    )

    return (
        mean_no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
        or high_no_speech_ratio >= NO_SPEECH_SEGMENT_RATIO_THRESHOLD
        or mean_avg_logprob <= LOW_CONFIDENCE_AVG_LOGPROB_THRESHOLD
        or (
            mean_no_speech_prob >= NO_SPEECH_WITH_LOW_CONFIDENCE_THRESHOLD
            and mean_avg_logprob <= -0.7
        )
    )


def _collect_float_metadata(items: list[Any], key: str) -> list[float]:
    values = []
    for item in items:
        value = metadata_value(item, key)
        if value is not None:
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio_at_least(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value >= threshold) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kvant voice assistant")
    parser.add_argument("--webhook", type=str, help="URL webhook для n8n")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader(CONFIG_PATH).load(webhook_override=args.webhook)
    assistant = VoiceAssistant(config)
    assistant.run()

if __name__ == "__main__":
    main()
