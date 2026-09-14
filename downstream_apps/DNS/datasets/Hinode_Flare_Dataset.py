import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset

class HinodeFlareDataset(HelioNetCDFDataset):
    """
    Downstream dataset built from ``Hinode_Flare_Catalogue_with_cells.csv``, following the
    same Surya-alignment pattern as ``FlareDSDataset`` in ``template_dataset.py``: each flare
    event is matched to the nearest Surya index timestep via ``pd.merge_asof``, and
    ``__getitem__`` returns the same shape of sample dict (``forecast`` / ``ds_index``, plus
    the base class's Surya stack when requested).

    The Hinode catalog differs from the generic flare index used by ``FlareDSDataset`` in two
    ways this class handles directly:
      - it has no pre-computed ``intensity`` column, only a ``"X-ray class"`` string
        (e.g. ``"C4.7"``), which is converted to physical flux via
        :func:`goes_class_to_intensity`.
      - its timestamp columns (``start``/``peak``/``end``) are ``m/d/yy H:mm`` strings with a
        two-digit year, parsed explicitly rather than relying on ISO inference.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (flare forecasting supplies its own
    labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned
            dict. Set to False to return only the flare intensity label (useful for label
            inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick
            experiments.
        label_transform: Optional callable applied to the flux-converted ``intensity`` column
            to produce the ``normalized_intensity`` label. Signature:
            ``(series: pd.Series) -> pd.Series``. If ``None``, the raw flux values (W/m^2) are
            used as-is.
        ds_flare_index_path: Path to ``Hinode_Flare_Catalogue_with_cells.csv``.
        ds_time_column: Column in the catalog to use as the event timestamp. One of
            ``"start"``, ``"peak"``, ``"end"``. Defaults to ``"start"`` so that, combined with
            ``ds_match_direction="forward"``, the matched Surya frame precedes the flare
            (causal prediction).
        ds_time_tolerance: Maximum allowed time offset when matching Surya and catalog
            timestamps (e.g. ``"4d"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).

    Raises:
        ValueError: If ``ds_flare_index_path`` is not provided, or if no overlap exists
            between the Surya and catalog indices within the specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        label_transform: Callable[[pd.Series], pd.Series] | None = None,
        ds_flare_index_path: str | None = None,
        ds_time_column: Literal["start", "peak", "end"] = "start",
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")
        if ds_time_column not in ["start", "peak", "end"]:
            raise ValueError("ds_time_column must be one of 'start', 'peak', or 'end'")

        # load_forecast_frames defaults to False here: flare forecasting supplies its
        # own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load the Hinode catalog and find intersection with Surya index
        if ds_flare_index_path is not None:
            self.ds_index = pd.read_csv(ds_flare_index_path)
        else:
            raise ValueError("ds_flare_index_path must be provided for HinodeFlareDataset")

        # Hinode timestamps are "m/d/yy H:mm" strings, e.g. "3/31/26 21:41".
        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column], format="%m/%d/%y %H:%M"
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Create Surya valid indices and find closest match to Hinode index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index, # ds_index is not an index, this is our CSV file
            right_on="ds_index", # This ds_index is the DateTime column of our CSV
            left_on="valid_indices", # This valid_indices is the DateTime of surya things
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        ) # Compute difference between surya dates and our dates
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and Hinode indices")

        # Override valid indices variables to reflect matches between Surya and Hinode
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None and max_number_of_samples < self.adjusted_length:
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Normalized flare intensity label (W/m^2, unless
                    ``label_transform`` was applied).
                ds_index (str): ISO-format timestamp from the Hinode flare index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        sample["forecast"] = self.df_valid_indices.iloc[idx]["cell_number"].astype(np.int8)
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample
