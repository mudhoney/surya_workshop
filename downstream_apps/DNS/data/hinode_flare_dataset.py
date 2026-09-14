import pandas as pd
from torch.utils.data import Dataset


class HinodeFlareCatalogueDataset(Dataset):
    """Dataset wrapping Hinode_Flare_Catalogue_with_cells.csv."""

    def __init__(self, csv_path: str = "Hinode_Flare_Catalogue_with_cells.csv"):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return self.df.iloc[idx].to_dict()
