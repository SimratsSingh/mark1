import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import pyttsx3
import tflearn
import tensorflow
import random
import json
import pickle
import speech_recognition as sr
from twilio.rest import Client

account_sid = 'AC99bcc9962dddbb5fe7526e66ff7b4e0d'
auth_token = '3d938290ea5ee13fcdeb8d2a32c50823'

client = Client(account_sid, auth_token)

r = sr.Recognizer()
with open("resource/intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickel","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk .word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickel","wb") as f:
        pickle.dump((words, labels, training, output),f)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bad_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def speaktext(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


def msg():

    message = client.messages.create(
        from_='+19252414468',
        body='someone is waiting for you in the lobby',
        to='+919871513233'
    )

    # print(message.sid)


def voicechat():
    print("START TALKING TO A BOT. TYPE 'QUIT' TO STOP")
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("you: " + MyText)
                if MyText == "quit":
                  break

                results = model.predict([bad_of_words(MyText, words)])[0]
                results_index = np.argmax(results)
                tag = labels[results_index]

                if results[results_index] > 0.8:
                    for tg in data["intents"]:
                        if tg["tag"] == tag:
                            responses = tg["responses"]

                    botans = random.choice(responses)
                    speaktext(botans)
                    if botans == "A message has been sent to him. Until he comes please wait":
                        msg()
                    print("bot: " + botans)
                else:
                    print("I'm sorry can't understand. Can you please repeat or try something else")
                    speaktext("I'm sorry can't understand. Can you please repeat or try something else")

        except sr.RequestError as e:
            print("Could not request results")
            speaktext("Could not request results")

        except sr.UnknownValueError:
            print("")
voicechat()

