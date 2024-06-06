# Searches the database using Rick Astley's "Never Gonna Give You Up" as an example

from fyp import SearchConfig, SongSearchState, search_song

def main():
    config = SearchConfig(
        # Refer to the comments in the SearchConfig class for more info
        max_transpose=3,
        chord_model_path="./resources/ckpts/btc_model_large_voca.pt",
        beat_model_path="./resources/ckpts/beat_transformer.pt",
        cache_dir="./resources/cache",
        # extra_info="{title}/{views}",
        verbose=True
    )

    state = SongSearchState(link="https://www.youtube.com/watch?v=dQw4w9WgXcQ", config=config)

    # Search the song
    scores = search_song(state=state)

if __name__ == "__main__":
    main()
