import subprocess
import sys

# install all the packages needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])

import re
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.svm import LinearSVC 
from fuzzywuzzy import fuzz
import os
import json
import time


user_details = {}
chatbot_name = "SpotBot"
songs_dataset = pd.read_csv("spotify_dataset.csv")

# what my dataset is
corpus = pd.read_csv("knowledge_dataset.csv")
corpus["Question"] = corpus["Question"].str.lower()

# what my intents are
data = {
    'intents': [
        {'intent': 'greet', 'examples': ['hello', 'hi', 'hey']},
        {'intent': 'create_playlist', 'examples': ['create a playlist', 'make a playlist', 'set up a playlist']},
        {'intent': 'rename_playlist', 'examples': ['rename a playlist', 'change a playlist name', 'change the name of a playlist']},
        {'intent': 'add_to_playlist', 'examples': ['add a song to my playlist', 'add a song to my playlist', 'add a song to my playlist']},
        {'intent': 'remove_from_playlist', 'examples': ['remove a song from my playlist', 'delete a song from my playlist', 'delete a song from my playlist']},
        {'intent': 'delete_playlist', 'examples': ['delete a playlist', 'remove a playlist', 'delete a playlist']},
        {'intent': 'bot_identity', 'examples': ['who are you', 'what is your name', "what's your name","who r u"]},
        {'intent': 'user_identity', 'examples': ['what is my name', "what's my name", "who am i", "what's my identity"]},
        {'intent': 'bot_capabilities', 'examples': ['what can you do', 'how can you help me']},
        {'intent': 'tell_story', 'examples' : ['tell me a story', 'tell a short story']},
        {'intent': 'small_talk', 'examples': ["how are you", "how r u", "hows your day", "how's the weather", "hows it g", "how's everything", "got plans for the weekend?", "how's it going", "wys g", "whatsup"]},
        {'intent': 'joke', 'examples': ['tell me a joke', 'tell a joke', 'say a joke']},
        {'intent': 'time', 'examples': ['what is the time', 'what time is it', 'what time is it now', 'what is the date', 'what is the day']},
        {'intent': 'question_data_set', 'examples':["how are glacier caves formed?","how are antibodies used in","what are stocks and bonds","what artist have song with ashanti?",
                                                    "How Works Diaphragm Pump","what happened to george o'malley on grey's anatomy?","what country is turkey in","what are use taxes?",
                                                    "who owns land rover","What U.S. President's head has been featured on the nickel (five-cent coin) since 1938?","who won fifa world cup 2010",
                                                    "what part of the government governs the US post office?","what are stanzas in poetry","what role do ombudsman play in the swedish government?",
                                                    "what year did mexico gain independence from spain","who killed robert kennedy","what makes a dwarf planet","what is the highest mountain in america and where is is located?"]},
    ]
}

# a story to tell
story = """
Once, at the crossroads of ancient maritime routes, there was a small but strategically located island known as Singapura, or "Lion City." According to legend, a Sumatran prince named Sang Nila Utama landed on the island in the 13th century and, seeing a majestic lion, considered it a good omen and founded a settlement there.

For centuries, Singapore remained a relatively obscure fishing village. Its fate changed dramatically in the early 19th century with the arrival of Sir Stamford Raffles, a British statesman seeking a new British base in the region. In 1819, he established Singapore as a trading post for the British East India Company. The new settlement quickly grew as a trading hub, attracting immigrants from China, India, the Malay Archipelago, and beyond.

Singapore's strategic location made it a prized possession and a crossroads of East and West. Over the years, it evolved into a bustling port city known for its diversity and economic importance. However, this prosperity was interrupted by World War II when Singapore fell to Japanese forces in 1942, a period marked by hardship and suffering.

After the war, a growing sense of national identity fueled aspirations for self-governance. Under the leadership of Lee Kuan Yew and the People's Action Party, Singapore embarked on a path toward independence. After a brief merger with Malaysia (1963-1965), Singapore became an independent republic on August 9, 1965.

Facing challenges like limited natural resources and a lack of hinterland, Singapore focused on developing its human capital and establishing itself as a global financial center. Through strategic planning, anti-corruption measures, and investment in public housing, healthcare, and education, Singapore transformed itself from a colonial outpost into one of the world's most prosperous nations.

Today, Singapore stands as a testament to the power of vision, planning, and hard work. It's a cosmopolitan city-state and a global financial hub, renowned for its skyscrapers, gardens, and a vibrant blend of cultures. The story of Singapore is one of resilience, transformation, and the unyielding spirit of a people determined to make their mark on the world.
"""

training_phrases = []
training_labels = []

for intent_record in data['intents']:
    for example in intent_record['examples']:
        training_phrases.append(example)
        training_labels.append(intent_record['intent'])


# using vectorizer to convert text to vector
all_text = corpus["Question"].tolist() + training_phrases
vectorizer = TfidfVectorizer()
vectorizer.fit(all_text)
question_vectors = vectorizer.transform(corpus["Question"])

x_training_phrases = vectorizer.transform(training_phrases)

classifier = LinearSVC()
classifier.fit(x_training_phrases, training_labels)

# predict intent of user input
def predict_intent(phrase):
    # vectorise the phrase
    phrase_vec = vectorizer.transform([phrase.lower()])
    # predict the intent
    predicted_intent = classifier.predict(phrase_vec)[0]

    return predicted_intent


# function to find closest question related to input
def find_closest_question(input_text, question_vectors, corpus, vectorizer, threshold=0.5):

    input_text_lower = input_text.lower()
    if input_text_lower in corpus["Question"].values:
        exact_match = corpus["Question"].tolist().index(input_text_lower)
        return corpus.iloc[exact_match]["Answer"]

    input_vec = vectorizer.transform([input_text.lower()])
    if input_vec.shape[1] != question_vectors.shape[1]:
        raise ValueError("Make sure your corpus and input vectors are of the same size!") 
    similarities = cosine_similarity(input_vec, question_vectors)
    closest_question_idex = similarities.argmax()
    max_similarity = similarities[0, closest_question_idex]


    if max_similarity < threshold:
        return "Sorry I don't understand that question, I am still learning!"
    return corpus.iloc[closest_question_idex]["Answer"]

# helper functions
def load_user_details():
    try:
        with open("user_details.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# helper functions
def save_user_details(user_details):
    with open("user_details.json", "w") as file:
        json.dump(user_details, file)

def generate_new_user_id():
    # generates a unique user ID using the current timestamp that stores the user name in user_details.json
    return f"user_{int(time.time())}"

# function to capture name if user says
def capture_name(text, user_details):

    global user_id

    name_patterns = [
        r"my name is ([\w\s]+)",
        r"i'm ([\w\s]+)",
        r"call me ([\w\s]+)",
        r"i am ([\w\s]+)"
    ]

    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip().title()
            if name in user_details:
                user_id = user_details[name]['user_id']
                return f"Chatbot: Welcome back, {name}! What can I do for you today?"
            else:
                # generates a unique user ID for each user
                user_id = generate_new_user_id()
                user_details[name] = {'user_id': user_id}
                save_user_details(user_details)
                return f"Chatbot: Nice to meet you, {name}! What can I do for you today?"

    if text.strip():
        name = text.strip().title()
        if name in user_details:
            user_id = user_details[name]['user_id']
            return f"Chatbot: Welcome back, {name}! What can I do for you today?"
        else:
            user_id = generate_new_user_id()
            user_details[name] = {'user_id': user_id}
            save_user_details(user_details)
            return f"Chatbot: Nice to meet you, {name}! What can I do for you today?"

    return "Chatbot: I'm sorry, I didn't catch that. What is your name?"

# function to get user name
def get_user_name(user_id, user_details):
    for name, id in user_details.items():
        if id['user_id'] == user_id:
            return name
    return "Unknown user!"

# handle small talk
def is_small_talk(text):
    small_talk_phrases = [
        "how are you", "how r u", "hows your day", "how's the weather",
        "hows it g", "how's everything", "got plans for the weekend?", 
        "how's it going,","wys g", "whatsup"]
    
    for phrase in small_talk_phrases:
        if fuzz.partial_ratio(text.lower(), phrase) > 80:
            return True
    return False

# generate a random small talk response based on input
def generate_small_talk_response(input_text):
    for key, responses in small_talk_responses.items():
        if fuzz.partial_ratio(input_text, key) > 80:
            return random.choice(responses)
    return "Chatbot: I'm not sure how to respond to that, tell me more?"


def chatbot_response(user_input):
    # first normalize user input to lowercase for matching to avoid error
    user_input_lower = user_input.lower()

    # checks if the input matches small talk patterns
    for key, responses in small_talk_responses.items():
        if isinstance(key, tuple):
            if any(sub_key in user_input_lower for sub_key in key):
                return random.choice(responses)
        elif key in user_input_lower:
            return random.choice(responses)

    # predicts intent for non-small talk inputs
    intent = predict_intent(user_input)
    
    if intent == 'small_talk':
        return generate_small_talk_response(user_input)
    
    # set a default response if no intent matches
    return "Chatbot: Sorry, I don't understand that yet! Can you tell me more?"

# dictionary of small talk responses
small_talk_responses = {
    ("hello", "hi", "hey"): [
        "Hi there! How can I help you today?",
        "Hello! Always here to chat if you need.",
        "Hey! What can I do for you?"
    ],

    "how are you": [
        "I'm just a bot, but I'm doing well! How about you?",
        "Feeling digital and functional! And you?",
        "I'm running smoothly! Hope you are too."
    ],

    "how's it going": [
        "Going great, thanks! What's up with you?",
        "All systems operational. How can I assist you?",
        "It's going well! Let me know how I can help you."
    ],

    "how's the weather": [
        "I'm not sure about the weather, but I hope it's pleasant where you are!",
        "Weather data not found, but I hope it's a good day for you!",
        "I can't check the weather, but I recommend a window for the best results!"
    ],
    "what's your name": [
        "My name is SpotBot!",
        "I'm SpotBot, nice to meet you!",
    ],
    "who are you":[
        "My name is SpotBot!",
        "I'm SpotBot, nice to meet you!",
    ]
}

# response to telling a user joke
def tell_joke():
    jokes = [
        "I wouldn't buy anything with velcro. It's a total rip-off.",
        "What kind of tea is hard to swallow? Reality...",
        "Why was the math book sad? It had too many problems.",
        "Why was the broom late? It overswept!",
        "What do you call a fly without wings? A walk."
    ]
    return random.choice(jokes)

# output date and time for time intent
def handle_date_and_time():
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%H:%M:%S")
    return date, time

# list of things that my bot can do
def list_capabilities():
    capabilities = [
        "I can tell you the date and time",
        "I can remember your name",
        "I can have small talk with you",
        "I can tell you a joke",
        "I can be your friend",
        "I can create a playlist for you",
        "I can add songs to your playlist",
        "I can remove songs from your playlist",
        "I can delete your playlist",
        "I can recommend songs to you",
        "I can rename your playlist",
        "I can tell you a story",
        "I can tell you about myself",
    ]

# randomly shuffle capabilities output to user
    random.shuffle(capabilities)
    n = 5
    return "Here's some things I can do!:\n" + "\n".join(capabilities[:5])

# function to get music preferences from user
def get_music_preferences():
    preferences = {
        'genre': '',
        'artist': ''
    }
    preferences["genre"] = input("Chatbot: What genre of music do you like? :  ").strip().lower()
    preferences["artist'"] = input("Chatbot: Any favourite artist you like? : ").strip().lower()
    playlist_size = input("Chatbot: How many songs would you like in your playlist? (Enter a number): ").strip()

    if playlist_size.isdigit() and int(playlist_size) > 0:
        preferences['playlist_size'] = int(playlist_size)
    else:
        print("Chatbot: Invalid input for playlist size.")

    return preferences

# function to create playlist
def create_playlist(preferences, songs_dataset):
    filtered_songs = songs_dataset
    genre_search = preferences["genre"].lower()

    if preferences["genre"]:
        # split the genres and strip spaces to match accurately
        filtered_songs = filtered_songs[filtered_songs["Genre"].apply(lambda x: genre_search in [genre.strip() for genre in str(x).lower().split(',')])]

    if preferences["artist"]:
        artist_search = preferences["artist"].lower()
        filtered_songs = filtered_songs[filtered_songs["Artist"].str.lower() == artist_search]

    shuffled_playlist = filtered_songs.sample(min(len(filtered_songs), preferences['playlist_size']))

    return shuffled_playlist

# function to save playlist
def save_playlist(playlist, default_name= "playlist"):
    playlist_count = 1
    playlist_name = default_name
    extension = ".txt"

    filepath = f"{playlist_name}{playlist_count}{extension}"
    while os.path.exists(filepath):
        playlist_count += 1
        filepath = f"{playlist_name}{playlist_count}{extension}"
        

    with open(filepath, "w") as file:
        for index, song in playlist.iterrows():
            file.write(f"{song['Song Name']} by {song['Artist']}\n")

    
    return filepath

# function to rename playlist
def rename_playlist(current_playlist = None):
    if current_playlist:
        current_name = current_playlist
    else:
        current_name = input("Chatbot: What is the current name of the playlist? (please exclude the .txt): ")
    new_name = input("Chatbot: What is the new name you want: ")

    if os.path.exists(f"{current_name}.txt"):
        os.rename(f"{current_name}.txt", f"{new_name}.txt")
        print("Chatbot: Playlist renamed successfully! What else can I help you with today?")
    else:
        print("Chatbot: Hmmm that playlist doesn't exist! Try another name!")

# function to add song to playlist
def add_to_playlist(playlist_name, song_dataset):
    if not os.path.exists(f"{playlist_name}.txt"):
        print(f"Chatbot: Sorry, {playlist_name} doesn't exist!")
        return
    
    add_which_song = input("Chatbot: What is the name of the song you want to add? : ").strip().lower()

    add_which_song_artist = input("Chatbot: What is the name of the artist? : ")


    
    song_being_added = songs_dataset[(songs_dataset["Song Name"].str.lower() == add_which_song) & 
                                     (songs_dataset["Artist"].str.lower().apply(lambda x: add_which_song_artist in str(x).lower().split(',')))]
    
    if song_being_added.empty:
        print(f"Chatbot: Sorry, {add_which_song} doesn't exist!")
        return
    with open(f"{playlist_name}.txt", "a") as file:
        for index, song in song_being_added.iterrows():
            file.write(f"{song['Song Name']} by {song['Artist']}\n")
        
    print(f"Chatbot: Added {add_which_song.title()} by {add_which_song_artist.title()} to {playlist_name}.txt! What else can I do for you today?")

# function to remove song from playlist
def remove_from_playlist(playlist_name, song_dataset):

    remove_which_song = input("Chatbot: What is the name of the song you want to remove? : ").strip().lower()

    if os.path.exists(f"{playlist_name}.txt"):
        with open(f"{playlist_name}.txt", "r") as file:
            lines = file.readlines()
            lines = [line for line in lines if remove_which_song not in line.lower()]
        with open(f"{playlist_name}.txt", "w") as file:
            file.writelines(lines)
        print(f"Chatbot: {remove_which_song.title()} has been removed from {playlist_name}. What else can I do for you today?")
    else:
        print("Chatbot: That playlist does not exist! Try again with another!")


def delete_playlist(playlist_name):
    if os.path.exists(f"{playlist_name}.txt"):
        os.remove(f"{playlist_name}.txt")
        print("Chatbot: Removed playlist successfully! What else can I help you with today?")
    else:
        print("Chatbot: That playlist does not exist! Try again with another!")

def get_time_greeting():
    current_hour = datetime.now().hour

    if 5<= current_hour < 12:
        return "Good morning!"
    if 12 <= current_hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"



# the main loop where bot runs
user_details = load_user_details()
prompt_name = True
user_id = None


# first prompts name
while prompt_name:
    time_greeting = get_time_greeting()
    input_text = input(f"Chatbot: {time_greeting}, I am your playlist chatbot SpotBot! What is your name? : ")
    name_response = capture_name(input_text, user_details)
    if name_response:
        print(name_response)
        prompt_name = False
        continue
    else:
        print("Chatbot: I'm sorry, I didn't catch that. What is your name?")
        continue

# all other operations happen in this loop
while True:
    input_text = input("You: ")

    if prompt_name:
        name_response = capture_name(input_text, user_details)
        if name_response:
            print(name_response)
            prompt_name = False
            continue
        else:
            print("Chatbot: I'm sorry, I didn't catch that. What is your name?")
            continue
    


    # to cut application from running
    if input_text.lower() in ["quit","exit", "stop", "bye"]:
        user_name = get_user_name(user_id, user_details)
        print(f"Chatbot: See you later {user_name}!")

        break

    intent = predict_intent(input_text)

    if is_small_talk(input_text):
        response = chatbot_response(input_text)
        print(f"Chatbot: {response}")
        continue


    elif intent == "bot_identity":
        print(f"Chatbot: My name is {chatbot_name}!")

        continue
    elif intent == "user_identity":
        print(f"Chatbot: Your name is {get_user_name(user_id,user_details)}!")

        continue
    elif intent == "bot_capabilities":
        print(f"Chatbot: {list_capabilities()}")

        continue

    elif intent == "joke":
        print(f"Chatbot: {tell_joke()}")

        continue

    elif intent == "tell_story":
        print("Chatbot" + story)

        continue

    elif intent == "greet":
        print(f"Chatbot: {chatbot_response(input_text)}")

        continue

    elif intent == 'time':
        date, time = handle_date_and_time()
        print(f"Chatbot: The date is {date} and the time is {time}")
        continue
    elif intent == 'question_data_set':
        response = find_closest_question(input_text, question_vectors, corpus, vectorizer)
        print(f"Chatbot: {response}")
        continue

    elif intent == 'create_playlist':
        print("Chatbot: Sure lets make you a playlist!")
        user_preferences = get_music_preferences()
        playlist = create_playlist(user_preferences, songs_dataset)
        user_name = get_user_name(user_id, user_details)

        print(f"Chatbot: Here is your playlist {user_name}!  I hope you like it!")
            
        saved_playlist = save_playlist(playlist)

        rename_choice = input("Chatbot: Do you want to name the playlist? (y/n): ")
        if rename_choice == 'y':
            rename_playlist(saved_playlist.replace(".txt", ""))
        else:
            print(f"Chatbot: Okay {user_name}, I have saved it as {saved_playlist}! What else can I do for you today?")
        continue

    elif intent == 'rename_playlist':
        rename_playlist()
        continue

    elif intent =='add_to_playlist':
        add_which_playlist= input("Chatbot: What is the name of the playlist? (please exclude the .txt): ").strip().lower()
        add_to_playlist(add_which_playlist, songs_dataset)

    elif intent == 'delete_playlist':
        playlist_name = input("Chatbot: What is the name of the playlist you want to delete? (please exclude the .txt): ")
        delete_playlist(playlist_name)
        continue
    
    elif intent == 'remove_from_playlist':
        remove_which_playlist = input("Chatbot: What is the name of the playlist you want to remove a song from? (please exclude the .txt): ").strip().lower()
        remove_from_playlist(remove_which_playlist, songs_dataset)
        continue

    else:
        print("Chatbot: Sorry, I don't understand that yet! Can you tell me more?")


