from __future__ import annotations

import concurrent.futures
import json
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, digest, file_hash, project_root, stable_seed, utcnow
from .market import equilibria

SOURCE_INFO = {
    "uci_retail_ii": {
        "page": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "citation": "Chen, D. Online Retail II. UCI Machine Learning Repository. DOI:10.24432/C5CG6D.",
        "terms": "CC BY 4.0",
    },
    "dominicks_oat": {
        "page": "https://www.chicagobooth.edu/research/kilts/research-data/dominicks",
        "citation": "Kilts Center for Marketing, University of Chicago Booth. Dominick's Finer Foods scanner data.",
        "terms": "Academic research use; see source terms. Raw records are not redistributed in this project.",
    },
}


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root() / p


def download(cfg: dict) -> dict:
    d = cfg["data"]
    root = resolve_path(d["cache_dir"])
    root.mkdir(parents=True, exist_ok=True)
    specs = {
        "uci_retail_ii.zip": d["uci_url"],
        "dominicks_oat.zip": d["dominicks_url"],
        "dominicks_upc.csv": d["dominicks_upc_url"],
    }
    lock_path = project_root() / "SOURCES.lock.json"
    packaged = json.loads(lock_path.read_text()) if lock_path.exists() else {}

    def one(item):
        name, url = item
        path = root / name
        expected = d.get("expected_sha256", {}).get(name) or packaged.get(name, {}).get("sha256")
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "TheseusCartel academic research"})
            try:
                with (
                    urllib.request.urlopen(request, timeout=d["download_timeout_seconds"]) as r,
                    temp.open("wb") as f,
                ):
                    shutil.copyfileobj(r, f)
                if path.suffix == ".zip" and not zipfile.is_zipfile(temp):
                    raise ValueError(f"Source returned non-zip content for {name}")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        actual = file_hash(path)
        if expected and expected != actual:
            raise ValueError(f"Source checksum mismatch for {name}; inspect upstream change before accepting")
        return name, {"url": url, "sha256": actual, "bytes": path.stat().st_size}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        manifest = dict(pool.map(one, specs.items()))
    atomic_json(root / "downloads.json", {"downloaded_utc": utcnow(), "files": manifest})
    return manifest


def decode_sas_hex(series: pd.Series) -> pd.Series:
    # Official CSV stores IEEE-754 doubles as big-endian hexadecimal strings.
    values = series.astype(str).str.strip()
    mapping = {s: struct.unpack(">d", bytes.fromhex(s))[0] for s in values.unique()}
    return values.map(mapping).astype(float)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["value"] = frame["price"] * frame["quantity"]
    frame["cost_value"] = frame["cost"] * frame["quantity"]
    group = frame.groupby(["product", "week"], observed=True, sort=True)
    panel = group.agg(
        quantity=("quantity", "sum"),
        value=("value", "sum"),
        cost_value=("cost_value", "sum"),
        rows=("quantity", "size"),
        description=("description", "first"),
    ).reset_index()
    panel["price"] = panel["value"] / panel["quantity"]
    panel["cost"] = panel["cost_value"] / panel["quantity"]
    if frame["cost"].isna().all():
        panel["cost"] = np.nan
    return panel


def process_uci(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        if len(members) != 1:
            raise ValueError("Unexpected UCI archive structure")
        sheets = pd.read_excel(z.open(members[0]), sheet_name=None, engine="openpyxl")
    raw = pd.concat(sheets.values(), ignore_index=True)
    before = len(raw)
    raw.columns = [str(c).strip().lower().replace(" ", "") for c in raw.columns]
    invoice = raw["invoice"].astype(str)
    code = raw["stockcode"].astype(str)
    ok = (
        (raw["quantity"] > 0)
        & (raw["price"] > 0)
        & ~invoice.str.upper().str.startswith("C")
        & code.str.fullmatch(r"[0-9]+[A-Za-z]*")
        & raw["description"].notna()
    )
    x = raw.loc[ok].copy()
    date = pd.to_datetime(x["invoicedate"], errors="coerce")
    x = x.loc[date.notna()]
    frame = pd.DataFrame(
        {
            "product": x["stockcode"].astype(str),
            "week": date.loc[x.index].dt.to_period("W").dt.start_time.astype(str),
            "price": x["price"].astype(float),
            "quantity": x["quantity"].astype(float),
            "cost": np.nan,
            "description": x["description"].astype(str).str.strip(),
        }
    )
    panel = _aggregate(frame)
    panel["dataset"] = "uci_retail_ii"
    return panel, {
        "input_rows": before,
        "valid_rows": len(frame),
        "cost_observed": False,
        "currency": "GBP",
        "excluded_customer_fields": True,
    }


def process_dominicks(path: Path, upc_path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Unexpected Dominick's movement archive")
        raw = pd.read_csv(z.open(members[0]), dtype={"PRICE_HEX": str, "PROFIT_HEX": str})
    raw.columns = raw.columns.str.strip().str.upper()
    meta = pd.read_csv(upc_path)
    meta.columns = meta.columns.str.strip().str.upper()
    price = decode_sas_hex(raw["PRICE_HEX"]) if "PRICE_HEX" in raw else raw["PRICE"]
    margin = decode_sas_hex(raw["PROFIT_HEX"]) if "PROFIT_HEX" in raw else raw["PROFIT"]
    # Codebook: sales = PRICE * MOVE / QTY. PROFIT is gross margin in percent,
    # implying average acquisition cost, NOT observed marginal/replacement cost.
    unit_price = price / raw["QTY"].replace(0, np.nan)
    cost = unit_price * (1 - margin / 100)
    ok = (
        (raw["MOVE"] > 0)
        & (raw["QTY"] > 0)
        & (unit_price > 0)
        & (cost > 0)
        & np.isfinite(unit_price)
        & np.isfinite(cost)
        & (raw["OK"] == 1)
    )
    desc = meta.drop_duplicates("UPC").set_index("UPC")["DESCRIP"]
    frame = pd.DataFrame(
        {
            "product": raw.loc[ok, "UPC"].astype(str),
            "week": raw.loc[ok, "WEEK"].astype(int),
            "price": unit_price[ok],
            "cost": cost[ok],
            "quantity": raw.loc[ok, "MOVE"].astype(float),
            "description": raw.loc[ok, "UPC"].map(desc).fillna("Oatmeal product").str.strip(),
        }
    )
    panel = _aggregate(frame)
    panel["dataset"] = "dominicks_oat"
    return panel, {
        "input_rows": len(raw),
        "valid_rows": len(frame),
        "cost_observed": True,
        "cost_interpretation": "Average acquisition cost used as a simulation cost proxy, not marginal cost",
        "currency": "USD",
        "full_precision_hex": "PRICE_HEX" in raw,
    }


def scenarios_from_panel(panel: pd.DataFrame, cfg: dict) -> tuple[list[dict], dict]:
    d, m = cfg["data"], cfg["market"]
    reference_nash = equilibria(
        m["a0"],
        tuple(m["qualities"]),
        tuple(m["normalized_costs"]),
        m["mu"],
        m["normalized_solver_tolerance"],
    )[0][0]
    scenarios, split_manifest = [], {}
    for dataset, dataset_frame in panel.groupby("dataset", sort=True):
        groups = []
        for product, group in dataset_frame.groupby("product", sort=True):
            if len(group) < d["minimum_product_weeks"] or group["rows"].sum() < d["minimum_product_rows"]:
                continue
            groups.append((str(product), group.sort_values("week")))
        groups.sort(key=lambda pair: stable_seed(cfg["seed"], "product-split", dataset, pair[0]))
        ndev, ntest = d["development_products_per_dataset"], d["test_products_per_dataset"]
        if len(groups) < ndev + ntest:
            raise ValueError(f"{dataset}: need {ndev + ntest} eligible products, have {len(groups)}")
        split_manifest[dataset] = {"eligible_products": len(groups), "development": [], "test": []}
        for i, (product, group) in enumerate(groups[: ndev + ntest]):
            split = "development" if i < ndev else "test"
            split_manifest[dataset][split].append(product)
            stop = int(np.floor(len(group) * d["temporal_fit_fraction"]))
            earlier, later = group.iloc[:stop], group.iloc[stop:]
            median_price = float(earlier["price"].median())
            volume = float(earlier["quantity"].median())
            observed_cost = earlier["cost"].notna().all()
            if observed_cost:
                alpha = median_price
                costs = [float(earlier["cost"].median() / alpha)] * 2
                cost_method = "recorded_average_acquisition_cost_proxy"
            else:
                alpha = median_price / reference_nash
                costs = list(m["normalized_costs"])
                cost_method = "structural_cost_assumption_from_original_game"
            if min(alpha, volume, *costs) <= 0:
                raise ValueError("Nonpositive scenario scale")
            scenario = {
                "id": dataset + "_" + product,
                "dataset": dataset,
                "product": product,
                "split": split,
                "description": str(group["description"].iloc[0]),
                "price_scale": alpha,
                "volume_scale": volume,
                "normalized_costs": costs,
                "cost_method": cost_method,
                "calibration_weeks": len(earlier),
                "heldout_weeks": len(later),
                "fit_last_week": str(earlier["week"].iloc[-1]),
                "heldout_first_week": str(later["week"].iloc[0]),
                "fit_median_price": median_price,
                "heldout_median_price": float(later["price"].median()),
                "demand_identified_from_transactions": False,
            }
            scenarios.append(scenario)
    return scenarios, split_manifest


def prepare(cfg: dict) -> Path:
    manifests = download(cfg)
    root = resolve_path(cfg["data"]["cache_dir"])
    out = resolve_path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(
        {
            "data_config": cfg["data"],
            "seed": cfg["seed"],
            "market": cfg["market"],
            "sources": manifests,
            "preprocessing_source_sha256": file_hash(Path(__file__)),
        }
    )
    meta_path = out / "manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") == fingerprint:
        return out / "scenarios.json"
    panels, cleaning = [], {}
    for name, loader, args in [
        ("uci_retail_ii", process_uci, [root / "uci_retail_ii.zip"]),
        ("dominicks_oat", process_dominicks, [root / "dominicks_oat.zip", root / "dominicks_upc.csv"]),
    ]:
        cache = out / (name + ".parquet")
        cleaning_file = out / (name + ".cleaning.json")
        source_hash = digest([file_hash(Path(__file__)), *[manifests[p.name]["sha256"] for p in args]])
        if cache.exists() and cleaning_file.exists():
            info = json.loads(cleaning_file.read_text())
            if info.get("source_sha256") == source_hash:
                p = pd.read_parquet(cache)
                panels.append(p)
                cleaning[name] = info
                continue
        p, info = loader(*args)
        info["source_sha256"] = source_hash
        p.to_parquet(cache, index=False)
        atomic_json(cleaning_file, info)
        panels.append(p)
        cleaning[name] = info
    scenarios, splits = scenarios_from_panel(pd.concat(panels, ignore_index=True), cfg)
    atomic_json(out / "scenarios.json", scenarios)
    atomic_json(
        meta_path,
        {
            "fingerprint": fingerprint,
            "sources": manifests,
            "source_info": SOURCE_INFO,
            "cleaning": cleaning,
            "splits": splits,
            "scenarios_sha256": digest(scenarios),
            "created_utc": utcnow(),
            "scope": "Observed retail contexts; counterfactual outcomes are simulated, not empirical collusion labels",
        },
    )
    return out / "scenarios.json"
