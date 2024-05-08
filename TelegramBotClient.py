import configparser
import os
import telebot
import util
from AudioChatClient import send_text
import io
from pydub import AudioSegment


util.load_secrets()

# Load settings from configuration file
config = configparser.ConfigParser()
config.read('config.ini')

if 'Settings' in config:
    SERVER_URL = config['Settings'].get('ServerAddress', 'http://localhost:5000')
    AGENT_NAME = config['Settings'].get('AgentName', 'Sophia')
else:
    SERVER_URL = 'http://localhost:5000'
    AGENT_NAME = 'Sophia'

BOT_TOKEN = os.environ.get('BOT_TOKEN')
TELEGRAM_CHAT_ID = int(os.environ.get('TELEGRAM_CHAT_ID'))

bot = telebot.TeleBot(BOT_TOKEN)


class IsAllowedUser(telebot.custom_filters.SimpleCustomFilter):
    # Class will check whether the user is admin or creator in group or not
    key = 'is_allowed_user'

    @staticmethod
    def check(message: telebot.types.Message):
        return message.from_user.id in [TELEGRAM_CHAT_ID]


# To register filter, you need to use method add_custom_filter.
bot.add_custom_filter(IsAllowedUser())


@bot.message_handler(commands=['start', 'hello'], is_allowed_user=True)
def send_welcome(message):
    response, audio = send_text("Hello!", SERVER_URL, AGENT_NAME, "User", audio_response=False)
    response_text = response['response']
    bot.send_message(message.chat.id, response_text)


@bot.message_handler(func=lambda msg: True, is_allowed_user=True)
def echo_all(message):
    bot.send_chat_action(chat_id=message.chat.id, action="typing")
    response, audio_data = send_text(message.text, SERVER_URL, AGENT_NAME,
                                     message.from_user.first_name, audio_response=True)
    response_text = response['response']

    # Convert audio data to AudioSegment
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # 16-bit samples
        frame_rate=24000,  # 24000 sample rate
        channels=1  # Mono
    )

    # Export audio as MP3
    mp3_bytes = io.BytesIO()
    audio_segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)

    # Send the text message and audio file
    bot.send_message(message.chat.id, response_text)
    bot.send_audio(message.chat.id, mp3_bytes)


print("Bot is running...")
bot.infinity_polling()
