import glob

import pandas as pd


def concat_all_results(outputs_filename="all_outputs.csv", files_regex="*_bs*.csv"):
    """Concatenates all files via regex to an output file"""
    candidate_files = glob.glob(files_regex)

    if candidate_files:
        df = pd.concat(
            [pd.read_csv(file_name, index_col=0) for file_name in candidate_files]
        ).reset_index(drop=True)
        df.to_csv(outputs_filename)
        print(f"Saved concatenated outputs to {outputs_filename}")

    else:
        print(f"No outputs to generate {outputs_filename}")


if __name__ == "__main__":
    concat_all_results()
