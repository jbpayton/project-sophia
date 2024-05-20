import configparser
import os
import discord
from discord.ext import commands
import io
from pydub import AudioSegment
from AudioChatClient import send_text
import util

# Load secrets (assuming util.load_secrets() is used to load environment variables)
util.load_secrets()

SEND_AUDIO = False

# Load settings from configuration file
config = configparser.ConfigParser()
config.read('config.ini')

if 'Settings' in config:
    SERVER_URL = config['Settings'].get('ServerAddress', 'http://localhost:5000')
    AGENT_NAME = config['Settings'].get('AgentName', 'Sophia')
else:
    SERVER_URL = 'http://localhost:5000'
    AGENT_NAME = 'Sophia'

DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
DISCORD_CHANNEL_ID = int(os.environ.get('DISCORD_CHANNEL_ID'))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Bot is running as {bot.user}')

def is_allowed_user(ctx):
    return ctx.channel.id == DISCORD_CHANNEL_ID

@bot.command(name='start')
@commands.check(is_allowed_user)
async def send_welcome(ctx):
    response, _ = send_text("Hello!", SERVER_URL, AGENT_NAME, "User", audio_response=False)
    response_text = response['response']
    await ctx.send(response_text)

@bot.command(name='hello')
@commands.check(is_allowed_user)
async def send_hello(ctx):
    response, _ = send_text("Hello!", SERVER_URL, AGENT_NAME, "User", audio_response=False)
    response_text = response['response']
    await ctx.send(response_text)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.id == DISCORD_CHANNEL_ID:
        await message.channel.typing()
        response, audio_data = send_text(message.content, SERVER_URL, AGENT_NAME,
                                         message.author.global_name, audio_response=SEND_AUDIO)
        response_text = response['response']

        if SEND_AUDIO:
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
        await message.channel.send(response_text)
        if SEND_AUDIO:
            await message.channel.send(file=discord.File(fp=mp3_bytes, filename="response.mp3"))

    await bot.process_commands(message)

bot.run(DISCORD_TOKEN)
