#This is an example callling JB hi-fi's customer support line.
#The institution starts by greeting you asking for your query.The option for the caller to upload an existing  Which is then transcribed and classified using sentiment analysis. The institution then 
#classifies your query to be directed to one of these departments Sales, Quality Assurance, Logistics, Finance, IT.The institution then asks for your confirmation
#to check if it has classified your query correctly, to which if given a positive sentiment response directs you to the confirmed department else prints 'CONFUSED'.
#Some of the intents it can read and handle is when a caller is asking for a financial, sales, IT, logistics, Quality Assurance related query.
#Some queries for the marker - 'The Iphone I ordered was delivered broken'
#                              'my order has not been delivered yet.'
#                              'I have not yet received the refund that was initiated'
#                              'Can I know the specifications of an Apple watch series 9.'
#                              'I am not able to login into my account.'
#                              'I am not able to add items into the cart.'


import transformers
from transformers import pipeline
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import soundfile as sf
import os
import logging

logging.basicConfig(filename='transcript.log', level=logging.INFO, format='%(asctime)s - %(message)s')


Sales = ['stock','status','colour','customisation','features','specs','specifications','discount','pricing']
Quality= ['broken','torn','replacement']
Tech = ['account','login','cart']
Fin = ['refund','payment','reimbursement','invoice','billing']
Logistics = ['delivery','delayed','return','status','customs','import','shipping','lost']

candidate_labels = ['stock','status','colour','customisation','features','specs','specifications','discount',
                    'pricing','delivery','delayed','return','status','customs','import','shipping','lost','account',
                               'login','cart','refund','payment','reimbursement','invoice','billing','broken','torn','replacement']

# Greeting customer
audio_file = '/home/abinand-balajee/project/greeting.wav'
data, samplerate = sf.read(audio_file)
sd.play(data, samplerate)
sd.wait()

userinp = input('Press R if you would like to record audio Or press F if you would like to load a file')
# Hearing query from cust
if userinp == 'R':
    duration = 5  
    fs = 44100  
    channels = 2  
    filename = "recorded_audio.wav"  
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait() 
    write(filename, fs, audio_data)
    print(f"Recording saved as {filename}")
elif userinp == 'F':
    filename = 'recorded_query.wav'
    if not os.path.isfile(audio_file):
        print("File not found.")
        exit()


#Transcribe query
model = whisper.load_model('small')
result = model.transcribe(filename)
print(result['text'])

logging.info(result['text'])

if result['text'] == '':
    print('CONFUSED')
    exit()



#Sentiment Analysis
senti = transformers.pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(senti(result['text']))

#Learning Intent
categories = transformers.pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
query = categories(result['text'],
           candidate_labels = ['stock','status','colour','customisation','features','specs','specifications','discount',
                               'pricing','delivery','delayed','return','status','customs','import','shipping','lost','account',
                               'login','cart','refund','payment','reimbursement','invoice','billing','broken','torn','replacement'],)
print(query['labels'])

if query['labels'][0] not in candidate_labels:
    print('CONFUSED')
    exit()


#Choosing department
if query['labels'][0] in Logistics:
    audiopath = '/home/abinand-balajee/project/logist_query.wav'
elif query['labels'][0] in Sales:
    audiopath = '/home/abinand-balajee/project/sales_query.wav'
elif query['labels'][0] in Fin:
    audiopath = '/home/abinand-balajee/project/fin_query.wav'
elif query['labels'][0] in Quality:
    audiopath = '/home/abinand-balajee/project/QA_query.wav'
elif query['labels'][0] in Tech:
    audiopath = '/home/abinand-balajee/project/IT_query.wav'

play,sample = sf.read(audiopath)
sd.play(play, sample)
sd.wait()


#Recording Confirmation
duration = 5  
fs = 44100  
channels = 2  
filename = "confirm_audio.wav"  
print("Recording...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
sd.wait()  
write(filename, fs, audio_data)
print(f"Recording saved as {filename}")

#Transcribe confirmation
model = whisper.load_model('small')
final = model.transcribe("confirm_audio.wav")
print(final['text'])

#confirmation sentiment
senti = transformers.pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
ment = senti(final['text'])
print(ment[0])


#Choosing confirm
if query['labels'][0] in Logistics and ment[0]['label'] == 'POSITIVE':
    path = '/home/abinand-balajee/project/logist_confirm.wav'
elif query['labels'][0] in Sales and ment[0]['label'] == 'POSITIVE':
    path = '/home/abinand-balajee/project/sales_confirm.wav'
elif query['labels'][0] in Fin and ment[0]['label'] == 'POSITIVE':
    path = '/home/abinand-balajee/project/fin_confirm.wav'
elif query['labels'][0] in Quality and ment[0]['label'] == 'POSITIVE':
    path = '/home/abinand-balajee/project/QA_confirm.wav'
elif query['labels'][0] in Tech and ment[0]['label'] == 'POSITIVE':
    path = '/home/abinand-balajee/project/IT_confirm.wav'
elif ment[0]['label'] == 'NEGATIVE':
    print('operator')
    exit()

# Continue if not confused
if 'path' in locals():
    pl,samp = sf.read(path)
    sd.play(pl,sample)
    sd.wait()
    exit()

