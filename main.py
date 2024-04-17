# Импортируем необходимые библиотеки
import torch
import telebot
import datetime
import requests
from multiprocessing import Process
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering,
                          pipeline,AutoModelForCausalLM, OPTModel)
#Загружаем обучение модели
model_path = 'modelforTechnoStrelka'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
text = "Как называется фильм в котором детей из детдомв набирают в группу для спец операций"
tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

label_ids = torch.argmax(probabilities, dim=1)


response = requests.get(
    'https://datasets-server.huggingface.co/rows?dataset=IlyaGusev%2Fru_turbo_saiga&config=default&split=train&offset=0&length=100')


df = []
js = response.json()
all_txt = ''
for i in range(len(js['rows'])):


    df.append(js['rows'][i]['row']['messages']['content'])



label = df[label_ids]

model_name = "timpal0l/mdeberta-v3-base-squad2"

model_answ = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer_answ = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline("question-answering", model=model_answ, tokenizer=tokenizer_answ)
import time
#Подключаем телеграм бота с помощью библиотек Telebot
bot = telebot.TeleBot('7099290086:AAEKR2cJPwgILCVr6J9yjrUUPt7UoUOVyxQ')
@bot.message_handler(commands=['start'])
#Функция старта
def start(message):
    a = open('listid.txt', 'a+', encoding='utf-8')
    a.seek(0)
    if str(message.chat.id) not in a.read():
        a.write(str(message.chat.id) + ' ')
        a.close()
    bot.send_message(message.chat.id, 'Доброго времени суток, '+ str(message.from_user.username))
    bot.send_message(message.chat.id, 'Задавайте вопрос, а я попытаюсь ответить на него максимально точно', )
@bot.message_handler(content_types=['text'])
#Принятие и обработка сообщений
def get_text_messages(message):
    user_input = message.text
    tokens = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    label_ids = torch.argmax(probabilities, dim=1)
    label = df[label_ids]
    QA_input = {
        'question': user_input,
        'context': label[0]
    }
    res = nlp(QA_input)
    bot.send_message(message.chat.id, f"{res['answer']}")
# Функция для рассылки термнов
def rass():
    b = open('термины.txt', 'r', encoding='utf-8')
    c = b.readlines()
    b.close()
    c = [i[:-1] for i in c if i[-1] == '\n']
    for i in range(len(c)-1):
        if c[i][-1] != '.':
            text = c[i]
            b = open('термины.txt', 'w', encoding='utf-8')
            t = c[:i] + ([c[i]+'.'])+c[i+1:]
            b.write('\n'.join(t))
            b.close()
            break
    else:
        text = c[0]
        for i in range(len(c)):
            c[i] = c[i][:-1]
        b = open('термины.txt', 'w', encoding='utf-8')
        b.write('\n'.join(c))
        b.close()
    f = open('listid.txt', 'r')
    listid = f.read().split()
    f.close()
    if datetime.datetime.now().strftime('%H.%M') == '12.20':
        for id in listid:
            bot.send_message(int(id), 'Сегодняшнее определение дня:')
            bot.send_message(int(id), text)
import time
def get_rass():
    while True:
        rass()
        time.sleep(60)
def start_bot2():
    bot.polling(none_stop=True, interval=0, timeout=60)
if __name__ == '__main__':
    while True:
        try:
            bot_thread1 = Process(target=get_rass, args=())
            bot_thread2 = Process(target=start_bot2, args=())
            bot_thread1.start()
            bot_thread2.start()
            bot_thread1.join()
            bot_thread2.join()
        except Exception as e:
            time.sleep(10)