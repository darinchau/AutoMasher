# Searches the database using Rick Astley's "Never Gonna Give You Up" as an example

from fyp import YouTubeURL, MashupConfig, mashup_song
from fyp.audio.cache import CacheHandler, LocalCache

def main():
    link = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    config = MashupConfig(
        starting_point=42.9, # seconds, next to chorus
        min_transpose=-3,
        max_transpose=3,
        _verbose=True
    )

    # Search the song
    audio = mashup_song(link, config, lambda url: LocalCache("./resources/cache", url))
    audio.save("output.wav")

if __name__ == "__main__":
    main()
