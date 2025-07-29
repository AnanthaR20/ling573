from glob import glob
import pandas as pd
import argparse
import os
import sys

def overwrite(data_dict):
    # TODO: overwrite or copy everything with patched indices to a new folder
    return

def analyze(data_dict):
    for key in data_dict.keys():
        print("##" * 15 + f" {key} " + "##" * 15)
        metadata = data_dict[key][1]
        print(metadata[["total_chunks", "num_targets"]].describe())
        print()

        # TODO: infer best k value range given statistics
    return

def load_data(data_dir_path):
    result = {}
    for filename in glob(f"{data_dir_path}/idx_*se3*.csv"):
        # Skip cases that aren't relevant for NLLP
        skip = "_toy" in filename or "_100" in filename or "_simple" in filename
        if skip:
            continue
        else:
            # We can read this metadata
            metadata = pd.read_csv(filename)

            # We can search for the corresponding data file
            data_filename = filename.replace("idx_", "")
            if os.path.exists(data_filename):
                data = pd.read_csv(data_filename)
            else:
                print(f"{data_filename} does NOT exist. Exiting early.")
                sys.exit(1)
            
            # Rename columns
            metadata = metadata.rename(
                columns={
                    "Unnamed: 0": "doc_id",
                    "idx": "total_chunks"
                }
            )

            # Explode, reset index, merge onto data DF
            metadata["chunk_idx"] = metadata.total_chunks.apply(lambda x: list(range(x)))
            metadata = metadata.explode("chunk_idx")
            metadata = metadata.reset_index()
            data["doc_id"] = metadata.doc_id
            data["chunk_idx"] = metadata.chunk_idx
            data = data.rename(
                columns={
                    "Unnamed: 0": "global_index"
                }
            )

            # One more metadata point: how many non-blank targets
            targets = []
            for _, group in data.groupby("doc_id"):
                targets.append(group.summary.notna().sum())

            metadata["num_targets"] = pd.Series(targets)

            # Prepare to update dictioanry
            key = filename.replace(f"{data_dir_path}/", "").replace(".csv", "").replace("billsum_clean_", "")
            result[key] = (data, metadata)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="specify data directory")
    args = parser.parse_args()

    # Get a dictionary with filenames as keys and tuples of (dataframe, count_dataframe) as values
    data_results = load_data(args.data_dir)

    # Log metadata
    analyze(data_results)

    # Overwrite old files
    # overwrite(data_results)

    return

if __name__ == "__main__":
    main()