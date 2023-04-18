import telegram
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import asyncio
import requests
import pymysql

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
    cursor = db.cursor()
    cursor.execute("use test")
    cursor.execute("insert into lab8_db (id, last_name, first_name, text) VALUES (%s, %s, %s, %s)" %\
                    (repr(update.message.chat.id), repr(update.message.chat.last_name),\
                     repr(update.message.chat.first_name), repr(update.message.text)))
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)
    cursor.execute("insert into lab8_db (id, last_name, text) VALUES (%s, %s, %s)" %\
                    (repr(update.message.chat.id), repr("bot"), repr(update.message.text)))
    db.commit()
    db.close()

async def cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
    cursor = db.cursor()
    cursor.execute("use test")
    cursor.execute("insert into lab8_db (id, last_name, first_name, text) VALUES (%s, %s, %s, %s)" %\
                    (repr(update.message.chat.id), repr(update.message.chat.last_name),\
                     repr(update.message.chat.first_name), repr(update.message.text)))
    db.commit()
    db.close()

    if update.message.text[1:5] == 'help':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Help!")
        db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
        cursor = db.cursor()
        cursor.execute("use test")
        cursor.execute("insert into lab8_db (id, last_name, text) VALUES (%s, %s, %s)" %\
                        (repr(update.message.chat.id), repr("bot"), repr("Help!")))
        db.commit()
        db.close()

    elif update.message.text[1:6] == 'visit':
        url = update.message.text.split(' ')[1]
        print(url)
        try:
            response = requests.get(url)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response.text)
            db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
            cursor = db.cursor()
            cursor.execute("use test")
            cursor.execute("insert into lab8_db (id, last_name, text) VALUES (%s, %s, %s)" %\
                            (repr(update.message.chat.id), repr("bot"), repr(response.text)))
            db.commit()
            db.close()

        except Exception as e:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=str(e))
            db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
            cursor = db.cursor()
            cursor.execute("use test")
            cursor.execute("insert into lab8_db (id, last_name, text) VALUES (%s, %s, %s)" %\
                            (repr(update.message.chat.id), repr("bot"), repr(str(e))))
            db.commit()
            db.close()
            print(e)

    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

if __name__ == '__main__':
    # bot = telegram.Bot("6152224720:AAGdDJiHJcycI2qg07y1qiHGSjwJHVOg1xs")
    application = ApplicationBuilder().token('6152224720:AAGdDJiHJcycI2qg07y1qiHGSjwJHVOg1xs').build()
    
    db = pymysql.connect(host="localhost", user="root", password="wljf1104", database="test", charset="utf8")
    cursor = db.cursor()
    cursor.execute("use test")
    cursor.execute("CREATE TABLE IF NOT EXISTS lab8_db(\
                   id VARCHAR(100) NOT NULL,\
                   last_name VARCHAR(20),\
                   first_name VARCHAR(20),\
                   text VARCHAR(1000) NOT NULL\
                   )")
    db.commit()
    db.close()

    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    cmd_handler = MessageHandler(filters.TEXT & filters.COMMAND, cmd)
    application.add_handler(echo_handler)
    application.add_handler(cmd_handler)

    application.run_polling()