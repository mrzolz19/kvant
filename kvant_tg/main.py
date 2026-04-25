#import Groq
import telebot
from groq import Groq
#import speech_recognition as sr
from os import path
import os
import glob
import ffmpeg
import requests
import uuid
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

load_dotenv()

bot_token = os.getenv("BOT_TELEGRAM_API")
if not bot_token:
    raise ValueError("Не задан BOT_TELEGRAM_API в .env")

bot = telebot.TeleBot(bot_token) # API Токен бота в телеграмме
session_id = str(uuid.uuid4())
webhook_n8n = os.getenv("WEBHOOK_N8N") # Вставьте сюда URL вашего webhook из n8n
owner_id_raw = os.getenv("OWNER_ID")
if not owner_id_raw:
    raise ValueError("Не задан OWNER_ID в .env")
if not webhook_n8n:
    raise ValueError("Не задан WEBHOOK_N8N в .env")
try:
    OWNER_ID = int(owner_id_raw)
except ValueError as exc:
    raise ValueError("OWNER_ID должен быть числом (Telegram user id)") from exc


def _http_fallback_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        return url
    return urlunparse(parsed._replace(scheme="http"))


def ai_response(user_message: str) -> str:
    data = {    
        "chatInput": user_message,
        "sessionId": session_id 
    }
    try:
        response = requests.post(webhook_n8n, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(result['output'])
        return result['output']

    except requests.exceptions.SSLError as ssl_err:
        # Частый кейс: endpoint работает по HTTP, а в .env указан HTTPS.
        if "WRONG_VERSION_NUMBER" in str(ssl_err).upper() and webhook_n8n.lower().startswith("https://"):
            fallback_url = _http_fallback_url(webhook_n8n)
            try:
                response = requests.post(fallback_url, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                print(result['output'])
                return result['output']
            except requests.exceptions.RequestException as fallback_err:
                print(f"Ошибка запроса к n8n (fallback HTTP): {fallback_err}")
                return "Извините, произошла ошибка при обработке запроса."
        print(f"SSL ошибка при запросе к n8n: {ssl_err}")
        return "Извините, произошла ошибка при обработке запроса."

    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к n8n: {e}")
        return "Извините, произошла ошибка при обработке запроса."

def audio_response(text: str) -> None:
    from speakerpy.lib_speak import Speaker as sp

    speaker = sp(model_id="v5_1_ru", language="ru", speaker="aidar", device="cpu")
    speaker.to_mp3(text=str(text), name_text="response", sample_rate=48000, audio_dir=".", put_accent=True, put_yo=True)

    # Найти сгенерированный файл (имя содержит хеш: out_response<hash>.mp3)
    mp3_files = glob.glob("out_response*.mp3")
    if not mp3_files:
        raise FileNotFoundError("Сгенерированный MP3 файл не найден")
    mp3_file = mp3_files[0]

    ffmpeg.input(mp3_file).output('output.ogg', format="ogg").overwrite_output().run()
    os.remove(mp3_file)

@bot.message_handler(content_types=['text'])
def text_processing(message: str) -> None:
    if message.from_user.id != OWNER_ID:
        return
    ai_reply_text = ai_response(message.text)
    bot.send_message(message.from_user.id, ai_reply_text)
    audio_response(ai_reply_text)
    with open('output.ogg', 'rb') as audio:
        bot.send_voice(message.from_user.id, audio)
    os.remove('output.ogg')

@bot.message_handler(content_types=['voice'])
def audio_processing(message):
    if message.from_user.id != OWNER_ID:
        return
    voice = message.voice
    print(voice)
    file_id = voice.file_id
    print(file_id)
    file_info = bot.get_file(file_id)
    print(file_info)
    downloaded_file = bot.download_file(file_info.file_path)

    with open('input.ogg', 'wb') as file:
        file.write(downloaded_file)
    ffmpeg.input('input.ogg').output('output.wav', format="wav").overwrite_output().run()
    audio_file = path.join(path.dirname(path.realpath(__file__)), 'output.wav')

    client = Groq()

    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_file, file.read()),
            model="whisper-large-v3",
            temperature=0,
            response_format="verbose_json",
        )
        recognized_text = transcription.text

    os.remove('input.ogg')
    os.remove('output.wav')

    # Отправляем распознанный текст через AI и возвращаем ответ пользователю
    ai_reply_text = ai_response(recognized_text)
    bot.send_message(message.from_user.id, ai_reply_text)
    audio_response(ai_reply_text)
    with open('output.ogg', 'rb') as audio:
        bot.send_voice(message.from_user.id, audio)
    os.remove('output.ogg')

bot.polling(none_stop=True, interval=0)


