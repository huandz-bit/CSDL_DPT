from src.search import search_voice

results = search_voice(
    "query1.wav"
)

for r in results:

    print(r)