# speech.py - MISO's Sprachverarbeitungsmodul

import openai
import sys
import os

# Stellt sicher, dass das Hauptverzeichnis im Suchpfad ist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Jetzt wird config.py korrekt geladen

def chat_with_miso(prompt):
    """Verarbeitet eine Benutzereingabe mit OpenAI GPT"""
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du bist MISO, ein hilfreicher KI-Assistent."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("MISO> ")
        if user_input.lower() in ["exit", "quit"]:
            print("MISO beendet.")
            break
        response = chat_with_miso(user_input)
        print(f"MISO: {response}")
print("MISO läuft erfolgreich!")


