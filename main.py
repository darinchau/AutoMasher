# Searches the database using Rick Astley's "Never Gonna Give You Up" as an example

from fyp import YouTubeURL, MashupConfig, mashup_song, MashupMode

def main():
    link = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    config = MashupConfig(
        # in seconds. Pick a complete verse, ideally at the start of the chorus, to get the best results
        starting_point=42.9,

        # The number of semitones to transpose the song
        # The song will be transposed by a random number of semitones between min_transpose and max_transpose
        min_transpose=-3,
        max_transpose=3,

        # The maximum and minimum relative BPM difference between the two songs
        # Say song A has 100 BPM, then song B can have a BPM between 100 * min_delta_bpm and 100 * max_delta_bpm
        # Anything outside of this range will be filtered out
        max_delta_bpm=1.25,
        min_delta_bpm=0.8,

        # The maximum "song distance" allowed between the two songs
        # Anything above this value will be filtered out
        # aka the larger this value, the more "worse" results the pipeline will return
        # See our paper for more information on song distance
        # Typically, a value between 3-5 will yield good results
        max_distance=4.5,

        # Filter only the best match from each song.
        # Say if song A matches with song B at both bar 8 with a score of 85 and bar 16 with a score of 90
        # If filter_first is True, the pipeline will only consider the match at bar 16
        # If filter_first is False, both results will be returned
        filter_first=True,

        # The range to perform beat extrapolation.
        # Keep at 3 unless you know what you're doing
        search_radius=3,

        # Keep only the top k results from the pipeline
        # instead of returning all results
        # This will make some parts slightly more efficient
        # but mostly it's for debugging purposes
        # Set to -1 to keep all results
        keep_first_k_results=10,

        # Filter out songs in the dataset that might have a faulty beat detection result
        # which is characterized by uneven bar lengths
        # This will also filter out songs that have drastic tempo changes
        filter_uneven_bars=True,
        filter_uneven_bars_min_threshold=0.9,
        filter_uneven_bars_max_threshold=1.1,

        # Filter out songs in the dataset that might have a faulty beat detection result
        # which is characterized by too few number of bars
        # This will filter out songs that has less than filter_short_song_bar_threshold bars
        filter_short_song_bar_threshold=12,

        # The mode to use when mashing up the songs
        # VOCALS_A will keep the vocals of song A and the music of song B
        # VOCALS_B will keep the vocals of song B and the music of song A
        # DRUMS_A will keep the drums of song A and the music of song B
        # DRUMS_B will keep the drums of song B and the music of song A
        # VOCALS_NATURAL will pick between VOCALS_A and VOCALS_B based on the activity of the vocals using heuristics below
        # DRUMS_NATURAL will pick between DRUMS_A and DRUMS_B based on the activity of the drums using heuristics below
        # NATURAL will pick between VOCALS_NATURAL and DRUMS_NATURAL based on the activity of the vocals and drums using heuristics below
        mashup_mode=MashupMode.NATURAL,

        # Threshold for heuristics detection. Not recommended to change
        natural_drum_activity_threshold=1,
        natural_drum_proportion_threshold=0.8,
        natural_vocal_activity_threshold=1,
        natural_vocal_proportion_threshold=0.8,
        natural_window_size=10,

        _verbose=True
    )

    # Search the song
    audio, scores, msg = mashup_song(link, config)
    print(msg)
    audio.save("output.wav")

if __name__ == "__main__":
    main()
