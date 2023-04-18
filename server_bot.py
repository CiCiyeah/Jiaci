import telegram
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import asyncio
import requests
import pymysql

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text[1:5] == 'help':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Help!")

    elif update.message.text[1:6] == 'visit':
        url = update.message.text.split(' ')[1]
        print(url)
        try:
            response = requests.get(url)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response.text)

        except Exception as e:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=str(e))
            print(e)

    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def run():
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    cmd_handler = MessageHandler(filters.TEXT & filters.COMMAND, cmd)
    application.add_handler(echo_handler)
    application.add_handler(cmd_handler)

if __name__ == '__main__':
    # bot = telegram.Bot("6152224720:AAGdDJiHJcycI2qg07y1qiHGSjwJHVOg1xs")
    application = ApplicationBuilder().token('6152224720:AAGdDJiHJcycI2qg07y1qiHGSjwJHVOg1xs').build()
    
    run()
    application.run_polling()