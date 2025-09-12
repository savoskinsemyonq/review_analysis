import pandas as pd
from datasets import load_dataset
from pathlib import Path

def main():
    dataset = load_dataset("imdb")

    Path("data").mkdir(exist_ok=True)

    all_dfs = []
    label_map = {0: "neg", 1: "pos"}

    for split in ["train", "test"]:
        df = pd.DataFrame(dataset[split])
        df["sentiment"] = df["label"].map(label_map)
        df = df.drop(columns=["label"])
        df["split"] = split
        df["review_id"] = [f"{split}_{i}" for i in range(len(df))]
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    out_path = "data/reviews.csv"
    full_df.to_csv(out_path, index=False)
    print(f"Saved {len(full_df)} reviews to {out_path}")

if __name__ == "__main__":
    main()
