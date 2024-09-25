#  The user should specify a YouTube link and a starting point in seconds to start their music mashup.
# The user will paste that into a text box.
# The UI should then display an YouTube embed that plays their song starting from the specified starting point.
# The user will then adjust some settings from a dropdown, and press a button to begin their mashup.
# A box will appear that prompts the user to wait for the mashup to complete.
# When the mashup has been completed, the user can click the mashup, listen to and download it.

# Internally its basically the main.py code but with a GUI wrapper around it.
# The GUI will be built using gradio

import gradio as gr
import traceback
from fyp.util import is_ipython, is_colab
from fyp.app import mashup_from_id
from fyp.audio.cache import LocalCache
from fyp import mashup_song, MashupConfig, MashupMode, YouTubeURL, InvalidMashup, Audio

# Set to None to disable caching
CACHE_DIR: str | None = "./resources/cache"

def get_mashup_mode_desc_map():
    return {
        MashupMode.VOCAL_A: "Vocal A",
        MashupMode.VOCAL_B: "Vocal B",
        MashupMode.DRUMS_A: "Drums A",
        MashupMode.DRUMS_B: "Drums B",
        MashupMode.VOCALS_NATURAL: "Vocal Auto",
        MashupMode.DRUMS_NATURAL: "Drums Auto",
        MashupMode.NATURAL: "Let the system decide",
    }

def pack_audio(audio: Audio):
    return gr.Audio((
        audio.sample_rate,
        audio.numpy()
    ))

def mashup(
    input_yt_link: str,
    starting_point: float,
    mashup_id: str,
    min_transpose_slider: int,
    max_transpose_slider: int,
    min_delta_bpm_slider: float,
    max_delta_bpm_slider: float,
    mashup_mode: str,
    max_distance_input: float,
    keep_first_k_input: int,
    filter_first: bool,
    filter_uneven_bars: bool,
    filter_uneven_bars_min_threshold_input: float,
    filter_uneven_bars_max_threshold_input: float,
    filter_short_song_bar_threshold_input: int,
    search_radius_input: int,
):
    cache_handler_factory = lambda url: LocalCache(CACHE_DIR, url)
    mashup_mode = MashupMode.NATURAL if not mashup_mode.strip() else {
        v: k for k, v in get_mashup_mode_desc_map().items()
    }[mashup_mode]

    config = MashupConfig(
        starting_point=starting_point,
        min_transpose=min_transpose_slider,
        max_transpose=max_transpose_slider,
        min_delta_bpm=min_delta_bpm_slider,
        max_delta_bpm=max_delta_bpm_slider,
        max_distance=max_distance_input,
        mashup_mode=mashup_mode,
        filter_first=filter_first,
        search_radius=search_radius_input,
        keep_first_k_results=keep_first_k_input,
        filter_uneven_bars=filter_uneven_bars,
        filter_uneven_bars_min_threshold=filter_uneven_bars_min_threshold_input,
        filter_uneven_bars_max_threshold=filter_uneven_bars_max_threshold_input,
        filter_short_song_bar_threshold=filter_short_song_bar_threshold_input,
        _verbose=True
    )

    if mashup_id and mashup_id != "Enter Mashup ID":
        try:
            mashup = mashup_from_id(mashup_id, config, cache_handler_factory)
            return gr.Textbox("Mashup complete!"), pack_audio(mashup)
        except InvalidMashup as e:
            print(traceback.format_exc())
            return gr.Textbox(f"Error: {e}"), None

    try:
        link = YouTubeURL(input_yt_link)
    except ValueError:
        return gr.Textbox("Error: Invalid YouTube link"), None

    try:
        mashup, _ = mashup_song(link, config, cache_handler_factory)
        return gr.Textbox("Mashup complete"), pack_audio(mashup)
    except Exception as e:
        print(traceback.format_exc())
        return gr.Textbox(f"Error: {e}"), None

def get_audio_from_link(input_yt_link: str, starting_point: float):
    try:
        link = YouTubeURL(input_yt_link)
    except ValueError:
        return gr.Textbox("Error: Invalid YouTube link"), None

    title = link.title
    cache_handler = LocalCache(CACHE_DIR, link)

    try:
        audio = cache_handler.get_audio()
        slice_end = min(audio.duration, starting_point + 10)
        audio = audio.slice_seconds(starting_point, slice_end)
    except Exception as e:
        print(traceback.format_exc())
        return gr.Textbox(f"Error: {e}"), None

    return gr.Textbox(title), pack_audio(audio)

def app():
    with gr.Blocks(title="AutoMasher") as app:
        gr.Markdown("## Auto Masher")
        gr.Markdown("""![Auto Masher](./resources/assets/auto_mashup.png)""")
        gr.Markdown(
            value="This software is provided for research and demo purposes only. All audio materials used in this pipeline constitutes as research, and thus, pursuant to Section 107 of the 1976 Copyright Act, constitutes as fair use. Please do not use it for commercial purposes."
        )
        with gr.TabItem("Create Mashup"):
            with gr.Row():
                with gr.Column():
                    input_yt_link = gr.Textbox(
                        label="Input YouTube Link",
                        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        interactive=True
                    )
                    starting_point = gr.Number(
                        label="Starting point (seconds)",
                        value=42.0,
                        interactive=True,
                        minimum=0.
                    )
                    title = gr.Textbox(
                        label="Video Title",
                        placeholder="Rick Astley - Never Gonna Give You Up",
                        interactive=False
                    )
                with gr.Column():
                    input_audio = gr.Audio(label="Input Audio")
                    refresh_button = gr.Button(
                        "Refresh input audio", variant="primary"
                    )
                    refresh_button.click(
                        get_audio_from_link,
                        [input_yt_link, starting_point],
                        [title, input_audio],
                    )
            with gr.Accordion("Advanced Settings"):
                with gr.Group():
                    with gr.Row(variant="panel"):
                        with gr.Column(scale=1.8): # type: ignore
                            with gr.Row():
                                min_transpose_slider = gr.Slider(
                                    label="Min Transpose",
                                    minimum=-6,
                                    maximum=6,
                                    value=-3,
                                    interactive=True,
                                )
                                max_transpose_slider = gr.Slider(
                                    label="Max Transpose",
                                    minimum=-6,
                                    maximum=6,
                                    value=3,
                                    interactive=True,
                                )
                            with gr.Row():
                                min_delta_bpm_slider = gr.Slider(
                                    label="Min Delta BPM",
                                    minimum=0.5,
                                    maximum=2,
                                    value=0.8,
                                    interactive=True,
                                )
                                max_delta_bpm_slider = gr.Slider(
                                    label="Max Delta BPM",
                                    minimum=0.5,
                                    maximum=2,
                                    value=1.25,
                                    interactive=True,
                                )
                            mashup_mode = gr.Radio(
                                label="Mashup Mode",
                                choices=(
                                    list(get_mashup_mode_desc_map().values())
                                ),
                                value="Vocal A",
                                interactive=True,
                            )
                            mashup_id = gr.Textbox(
                                label="Mashup ID",
                                placeholder="Enter Mashup ID"
                            )
                        with gr.Column():
                            with gr.Row():
                                max_distance_input = gr.Number(
                                    label="Max Song Distance",
                                    interactive=True,
                                    value=4.5,
                                    minimum=0
                                )
                                keep_first_k_input = gr.Number(
                                    label="Keep first k results",
                                    interactive=True,
                                    value=5,
                                )
                            with gr.Row():
                                filter_first = gr.Checkbox(
                                    label="Filter First", interactive=True
                                )
                                filter_uneven_bars = gr.Checkbox(
                                    label="Filter Uneven Bars",
                                    interactive=True,
                                    value=True
                                )
                            with gr.Row():
                                filter_uneven_bars_min_threshold_input = gr.Number(
                                    label="Min Threshold",
                                    interactive=True,
                                    value=0.9
                                )
                                filter_uneven_bars_max_threshold_input = gr.Number(
                                    label="Max Threshold",
                                    interactive=True,
                                    value=1.1
                                )
                            with gr.Row():
                                filter_short_song_bar_threshold_input = gr.Number(
                                    label="Short Song Bar Threshold",
                                    interactive=True,
                                    value=12
                                )
                                search_radius_input = gr.Number(
                                    label="Search Radius",
                                    interactive=True,
                                    value=3
                                )

            with gr.Group():
                with gr.Column():
                    mashup_button = gr.Button("Mashup!", variant="primary")
                    with gr.Row():
                        output_msg = gr.Textbox(label="Output Message")
                        output_audio = gr.Audio(
                            label="Output Audio (Click the icon with 3 dots to download)",
                        )

                    mashup_button.click(
                        mashup,
                        [
                            input_yt_link,
                            starting_point,
                            mashup_id,
                            min_transpose_slider,
                            max_transpose_slider,
                            min_delta_bpm_slider,
                            max_delta_bpm_slider,
                            mashup_mode,
                            max_distance_input,
                            keep_first_k_input,
                            filter_first,
                            filter_uneven_bars,
                            filter_uneven_bars_min_threshold_input,
                            filter_uneven_bars_max_threshold_input,
                            filter_short_song_bar_threshold_input,
                            search_radius_input,
                        ],
                        [output_msg, output_audio],
                    )

        if is_colab():
            app.queue(max_size=1022).launch(share=True)
        else:
            app.queue(max_size=1022).launch(
                server_name="0.0.0.0",
                inbrowser=True,
                server_port=8123,
                quiet=True,
            )

    return app


demo = app()
demo.launch(debug=True)
