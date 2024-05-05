## Code for producing the filtered dataset from the raw dataset

from datasets import Dataset, load_from_disk
from fyp.audio.search.align import filter_dataset, mapping_dataset

if __name__ == "__main__":
    dataset = load_from_disk("../features/audio-infos", keep_in_memory=True)
    dataset = dataset.filter(filter_dataset)
    dataset = dataset.map(mapping_dataset)
    dataset.save_to_disk("../features/audio-infos-filtered")
