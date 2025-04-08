sentence_pairs = [
    ("Hello, how are you?", "Hallo, wie geht es dir?"),
    ("Good morning.", "Guten Morgen."),
    ("I love learning new things.", "Ich liebe es, neue Dinge zu lernen."),
    ("The weather is nice today.", "Das Wetter ist heute schön."),
    ("This is a test sentence.", "Dies ist ein Testsatz."),
    ("I am a student.", "Ich bin ein Student."),
    ("My name is John.", "Mein Name ist John."),
    ("I like to read books.", "Ich lese gerne Bücher."),
    ("We are going to the park.", "Wir gehen in den Park."),
    ("Can you help me?", "Kannst du mir helfen?"),
    ("Where is the nearest restaurant?", "Wo ist das nächste Restaurant?"),
    ("I would like a cup of coffee.", "Ich hätte gerne eine Tasse Kaffee."),
    ("Do you speak English?", "Sprichst du Englisch?"),
    ("I am learning German.", "Ich lerne Deutsch."),
    ("She is my friend.", "Sie ist meine Freundin."),
    ("He is very kind.", "Er ist sehr nett."),
    ("The book is on the table.", "Das Buch liegt auf dem Tisch."),
    ("I need to go to the store.", "Ich muss zum Laden gehen."),
    ("How much does it cost?", "Wie viel kostet es?"),
    ("I am hungry.", "Ich habe Hunger."),
    ("Let's go to the cinema.", "Lass uns ins Kino gehen."),
    ("I enjoy playing soccer.", "Ich spiele gerne Fußball."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("I am tired.", "Ich bin müde."),
    ("We have a meeting tomorrow.", "Wir haben morgen ein Treffen."),
    ("What time is it?", "Wie spät ist es?"),
    ("Please, sit down.", "Bitte, setz dich."),
    ("I am very happy today.", "Ich bin heute sehr glücklich."),
    ("Do you want to join us?", "Willst du dich uns anschließen?"),
    ("Goodbye and see you soon.", "Auf Wiedersehen und bis bald.")
]

import random

random.seed(42)
# Path to the dataset1 file
input_file = r"english-to-german\deu.txt"

# List to store sentence pairs
sentence_pairs2 = []

# Open and process each line
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        # Split line by tab
        parts = line.strip().split('\t')
        
        # Only take lines that have at least two parts (English and German)
        if len(parts) >= 2:
            english = parts[0].strip()
            german = parts[1].strip()
            sentence_pairs2.append((english, german))


# Sample 10,000 random pairs if enough data exists
sample_size = min(100000, len(sentence_pairs2))
sampled_pairs = random.sample(sentence_pairs2, sample_size)


# Optional: print a sample
# for pair in sampled_pairs[:5]:
#     print(pair)
    
#print(len(sampled_pairs))
