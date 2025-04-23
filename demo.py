#  The user should specify a YouTube link and a starting point in seconds to start their music mashup.
# The user will paste that into a text box.
# The UI should then display an YouTube embed that plays their song starting from the specified starting point.
# The user will then adjust some settings from a dropdown, and press a button to begin their mashup.
# A box will appear that prompts the user to wait for the mashup to complete.
# When the mashup has been completed, the user can click the mashup, listen to and download it.

# Internally its basically the main.py code but with a GUI wrapper around it.
# The GUI will be built using gradio

import os
import base64
import gradio as gr
import traceback
from fyp.util import is_ipython, is_colab
from fyp.app import mashup_from_id, load_dataset
from fyp import mashup_song, MashupConfig, MashupMode, get_url, InvalidMashup, Audio

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
        int(audio.sample_rate),
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
    left_pan: float,
    save_original: bool,
    append_song_to_dataset: bool,
    dataset_path: str
):
    mashup_mode_ = MashupMode.NATURAL if not mashup_mode.strip() else {
        v: k for k, v in get_mashup_mode_desc_map().items()
    }[mashup_mode]

    config = MashupConfig(
        starting_point=starting_point,
        min_transpose=min_transpose_slider,
        max_transpose=max_transpose_slider,
        min_delta_bpm=min_delta_bpm_slider,
        max_delta_bpm=max_delta_bpm_slider,
        max_distance=max_distance_input,
        mashup_mode=mashup_mode_,
        filter_first=filter_first,
        search_radius=search_radius_input,
        keep_first_k_results=keep_first_k_input,
        filter_uneven_bars=filter_uneven_bars,
        filter_uneven_bars_min_threshold=filter_uneven_bars_min_threshold_input,
        filter_uneven_bars_max_threshold=filter_uneven_bars_max_threshold_input,
        filter_short_song_bar_threshold=filter_short_song_bar_threshold_input,
        left_pan=left_pan,
        _verbose=True,
        save_original=save_original,
        append_song_to_dataset=append_song_to_dataset,
        load_on_the_fly=False,
        assert_audio_exists=False,
        dataset_path=dataset_path,
    )

    if mashup_id and mashup_id != "Enter Mashup ID":
        try:
            mashup = mashup_from_id(mashup_id, config)
            return gr.Textbox("Mashup complete!"), pack_audio(mashup)
        except InvalidMashup as e:
            print(traceback.format_exc())
            return gr.Textbox(f"Error: {e}"), None

    try:
        link = get_url(input_yt_link)
    except Exception as e:
        return gr.Textbox(f"Error: Invalid YouTube link ({e})"), None

    try:
        mashup, _, system_message = mashup_song(link, config)
        return gr.Textbox(system_message), pack_audio(mashup)
    except Exception as e:
        print(traceback.format_exc())
        return gr.Textbox(f"Error: {e}"), None


def get_audio_from_link(input_yt_link: str, starting_point: float, dataset_path: str):
    try:
        link = get_url(input_yt_link)
    except Exception as e:
        return gr.Textbox(f"Error: Invalid YouTube link ({e})"), None

    title = link.video_title
    # Set load_on_the_fly to True to avoid loading the entire dataset into memory
    dataset = load_dataset(MashupConfig(1, dataset_path=dataset_path, load_on_the_fly=True))

    try:
        audio = dataset.get_audio(link)
        slice_end = min(audio.duration, starting_point + 10)
        audio = audio.slice_seconds(starting_point, slice_end)
    except Exception as e:
        print(traceback.format_exc())
        return gr.Textbox(f"Error: {e}"), None

    return gr.Textbox(title), pack_audio(audio)


def get_base64_logo():
    with open("resources/assets/auto_mashup.png", "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def app():
    with gr.Blocks(title="AutoMasher", css=".center-logo{margin: auto}") as app:
        gr.Markdown("## Auto Masher")
        gr.HTML(f"""<img class="center-logo" src="{get_base64_logo()}" alt="Auto Masher", width=150px height=150px>""")
        gr.Markdown(
            value="This demo is provided for research purposes only. All audio materials used in this pipeline constitutes as research, and thus, pursuant to Section 107 of the 1976 Copyright Act, constitutes as fair use. Please do not use it for commercial purposes."
        )
        with gr.TabItem("Create Mashup"):
            with gr.Row():
                with gr.Column():
                    input_yt_link = gr.Textbox(
                        label="Input YouTube Link",
                        value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        interactive=True,
                        info="Paste a YouTube link here to get started",
                    )
                    starting_point = gr.Number(
                        label="Starting point (seconds)",
                        value=43.0,
                        interactive=True,
                        minimum=0,
                        info="Pick a complete verse, ideally at the start of the chorus, to get the best results",
                    )
                    dataset_path = gr.Textbox(
                        label="Dataset Path",
                        interactive=True,
                        value="./resources/dataset",
                        info="The path to the dataset."
                    )
                with gr.Column():
                    title = gr.Textbox(
                        label="Video Title",
                        placeholder="Rick Astley - Never Gonna Give You Up",
                        interactive=False,
                        info="The title of the video will be displayed here",
                    )
                    input_audio = gr.Audio(label="Input Audio")
                    refresh_button = gr.Button(
                        "Refresh input audio", variant="primary",
                    )
                    refresh_button.click(
                        get_audio_from_link,
                        [input_yt_link, starting_point, dataset_path],
                        [title, input_audio],
                    )
            with gr.Accordion("Advanced Settings"):
                with gr.Group():
                    with gr.Row(variant="panel"):
                        with gr.Column(scale=1.3):  # type: ignore
                            with gr.Row():
                                min_transpose_slider = gr.Slider(
                                    label="Min Transpose",
                                    minimum=-6,
                                    maximum=6,
                                    value=-3,
                                    interactive=True,
                                    step=1,
                                    info="The minimum number of semitones to transpose the song. The search result will include songs transposed between min_transpose and max_transpose",
                                )
                                max_transpose_slider = gr.Slider(
                                    label="Max Transpose",
                                    minimum=-6,
                                    maximum=6,
                                    value=3,
                                    interactive=True,
                                    step=1,
                                    info="The maximum number of semitones to transpose the song. The search result will include songs transposed between min_transpose and max_transpose",
                                )
                            with gr.Row():
                                min_delta_bpm_slider = gr.Slider(
                                    label="Min Delta BPM",
                                    minimum=0.5,
                                    maximum=2,
                                    value=0.8,
                                    interactive=True,
                                    info="The minimum relative BPM difference between the two songs. Say song A has 100 BPM, then song B can have a BPM between 100 * min_delta_bpm and 100 * max_delta_bpm"
                                )
                                max_delta_bpm_slider = gr.Slider(
                                    label="Max Delta BPM",
                                    minimum=0.5,
                                    maximum=2,
                                    value=1.25,
                                    interactive=True,
                                    info="The maximum relative BPM difference between the two songs. Say song A has 100 BPM, then song B can have a BPM between 100 * min_delta_bpm and 100 * max_delta_bpm"
                                )
                            mashup_mode = gr.Radio(
                                label="Mashup Mode",
                                choices=(
                                    list(get_mashup_mode_desc_map().values())
                                ),
                                value="Let the system decide",
                                interactive=True,
                                info="The mode to use when mashing up the songs. "
                                "Vocals A will keep the vocals of song A and the music of song B. Vocals B will keep the vocals of song B and the music of song A"
                                "DRUMS_A will keep the drums of song A and the music of song B. DRUMS_B will keep the drums of song B and the music of song A. "
                                "VOCALS_NATURAL will pick between VOCALS_A and VOCALS_B based on the activity of the vocals using some heuristics"
                                "DRUMS_NATURAL will pick between DRUMS_A and DRUMS_B based on the activity of the drums using heuristics below."
                                "NATURAL will pick between VOCALS_NATURAL and DRUMS_NATURAL based on the activity of the vocals and drums using some heuristics"
                            )
                            mashup_id = gr.Textbox(
                                label="Mashup ID",
                                placeholder="Enter Mashup ID",
                                interactive=True,
                                info="If you have a mashup ID, you can enter it here to recreate the mashup. In this case the song link and starting point will be ignored"
                            )
                            with gr.Row():
                                with gr.Column():
                                    save_original = gr.Checkbox(
                                        label="Save Original",
                                        interactive=True,
                                        value=False,
                                        info="Save the original song in the output as well"
                                    )
                                    append_song_to_dataset = gr.Checkbox(
                                        label="Append Song to Dataset",
                                        interactive=True,
                                        value=True,
                                        info="Append the song to the dataset for future use"
                                    )
                                left_pan = gr.Number(
                                    label="Left Pan",
                                    interactive=True,
                                    value=0.15,
                                    minimum=-0.5,
                                    maximum=0.5,
                                    info="The left pan of the vocals in the output mashup. Other parts will be panned accordingly"
                                )
                        with gr.Column():
                            with gr.Row():
                                max_distance_input = gr.Number(
                                    label="Max Song Distance",
                                    interactive=True,
                                    value=8,
                                    minimum=0,
                                    info="The maximum 'song distance' allowed between the two songs. Anything above this value will be filtered out. Typically, a value around 5 will yield good results, and a value around 8 will have a more expansive collection of okayish results. See our paper for more information on song distance"
                                )
                                keep_first_k_input = gr.Number(
                                    label="Keep first k results",
                                    interactive=True,
                                    value=5,
                                    minimum=-1,
                                    info="Keep only the top k results from the pipeline instead of returning all results. This will make some parts slightly more efficient but mostly it's for debugging purposes. Set to -1 to keep all results"
                                )
                            with gr.Row():
                                filter_first = gr.Checkbox(
                                    label="Filter First",
                                    interactive=True,
                                    value=True,
                                    info="Filter only the best match from each song. Say if song A matches with song B at both bar 8 with a score of 85 and bar 16 with a score of 90. If filter_first is True, the pipeline will only consider the match at bar 16. If filter_first is False, both results will be returned"
                                )
                                filter_uneven_bars = gr.Checkbox(
                                    label="Filter Uneven Bars",
                                    interactive=True,
                                    value=True,
                                    info="Filter out songs in the dataset that might have a faulty beat detection result which is characterized by uneven bar lengths. This will also filter out songs that have drastic tempo changes"
                                )
                            with gr.Row():
                                filter_uneven_bars_min_threshold_input = gr.Number(
                                    label="Min Threshold",
                                    interactive=True,
                                    value=0.9,
                                    info="Filter out songs in the dataset that might have a faulty beat detection result which is characterized by uneven bar lengths. This will also filter out songs that have drastic tempo changes"
                                )
                                filter_uneven_bars_max_threshold_input = gr.Number(
                                    label="Max Threshold",
                                    interactive=True,
                                    value=1.1,
                                    info="Filter out songs in the dataset that might have a faulty beat detection result which is characterized by uneven bar lengths. This will also filter out songs that have drastic tempo changes"
                                )
                            with gr.Row():
                                filter_short_song_bar_threshold_input = gr.Number(
                                    label="Short Song Bar Threshold",
                                    interactive=True,
                                    value=12,
                                    info="Filter out songs in the dataset that might have a faulty beat detection result which is characterized by too few number of bars. This will filter out songs that has less than filter_short_song_bar_threshold bars"
                                )
                                search_radius_input = gr.Number(
                                    label="Search Radius",
                                    interactive=True,
                                    value=3,
                                    info="The range to perform beat extrapolation. Keep at 3 unless you know what you're doing"
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
                            left_pan,
                            save_original,
                            append_song_to_dataset,
                            dataset_path
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
demo.launch(debug=True, allowed_paths=[os.path.abspath("resources/assets")])
