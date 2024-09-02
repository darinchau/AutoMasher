# Searches the database using Rick Astley's "Never Gonna Give You Up" as an example

from fyp import SearchConfig, SongSearchState, search_song, YouTubeURL

def main():
    config = SearchConfig(
        # Refer to the comments in the SearchConfig class for more info
        max_transpose=3,
        chord_model_path="./resources/ckpts/btc_model_large_voca.pt",
        beat_model_path="./resources/ckpts/beat_transformer.pt",
        cache_dir="./resources/cache",
        dataset_path="./resources/dataset/audio-infos-v2.1.db",
        verbose=True,
        bar_number=20,
        nbars=8,
    )

    state = SongSearchState(link=YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), config=config)

    # Search the song
    scores = search_song(state=state)

    print("Scores:")
    print(scores)

if __name__ == "__main__":
    main()
