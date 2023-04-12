import os, requests
import gradio as gr
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
from dotenv import load_dotenv
load_dotenv()


messages = [{"role": "system", "content": 'you are a spanish speaking tutor.speak to me in Spanish only, and correct my grammer, text if needed. Respond to all input in 50 words or less. Speak in the first person. Do not use quotation marks. Do not say you are an AI language model.'}]

# prepare Q&A embeddings dataframe
# question_df = pd.read_csv('data/questions_with_embeddings.csv')
# question_df['embedding'] = question_df['embedding'].apply(eval).apply(np.array)

def transcribe(audio):
    global messages, question_df

    openai.api_key = os.getenv("OPENAI_KEY")
    # API now requires an extension so we will rename the file
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # question_vector = get_embedding(transcript['text'], engine='text-embedding-ada-002')

    # question_df["similarities"] = question_df['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
    # question_df = question_df.sort_values("similarities", ascending=False)

    # best_answer = question_df.iloc[0]['answer']

    user_text = f"{transcript['text']}" 
    messages.append({"role": "user", "content": user_text})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]
    print(system_message)
    messages.append(system_message)

    # text to speech request with eleven labs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.getenv('ADVISOR_VOICE_ID')}/stream"
    data = {
        "text": system_message["content"].replace('"', ''),
        "voice_settings": {
            "stability": 0.1,
            "similarity_boost": 0.8
        }
    }

    r = requests.post(url, headers={'xi-api-key': os.getenv("ELEVEN_LABS_KEY")}, json=data)

    output_filename = "reply.mp3"
    with open(output_filename, "wb") as output:
        output.write(r.content)

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    # return chat_transcript
    return chat_transcript, output_filename


# set a custom theme
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)

with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input
    # advisor = gr.Image(value=config.ADVISOR_IMAGE).style(width=config.ADVISOR_IMAGE_WIDTH, height=config.ADVISOR_IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio 
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True) 