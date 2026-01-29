from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt

from msu_miniseed import parse_file

def main() -> None:
    matplotlib.use("Agg")
    base = Path("input")
    paths = sorted(base.rglob("*.mseed"))
    if not paths:
        print("No .mseed files found under input/")
        return

    graphs_root = Path("output/graph")
    csv_root = Path("output/csv/")

    for path in paths:
        parsed = parse_file(path)
        df = parsed.dataframe
        print(f"{path}:")
        print(f"  records: {parsed.number_of_records}")
        print(f"  samples: {len(df)}")
        print(f"  first: {parsed.first_timestamp}")
        print(f"  last: {parsed.last_timestamp}")
        if not df.empty:
            min_val = df["sample"].min()
            max_val = df["sample"].max()
            mean_val = df["sample"].mean()
            median_val = df["sample"].median()
            print(f"  min sample: {min_val}")
            print(f"  max sample: {max_val}")

            rel = path.relative_to(base)
            out_png = graphs_root / rel.with_suffix(".png")
            out_svg = graphs_root / rel.with_suffix(".svg")
            out_csv = csv_root / rel.with_suffix(".csv")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            out_csv.parent.mkdir(parents=True, exist_ok=True)


            span_seconds = 0.0
            if len(df) > 1:
                span_seconds = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()

            fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
            ax.plot(df["timestamp"], df["sample"], color="black", linewidth=0.6)
            ax.axhline(min_val, color="tab:blue", linestyle="--", label=f"min={min_val:.6g}")
            ax.axhline(max_val, color="tab:red", linestyle="--", label=f"max={max_val:.6g}")
            ax.axhline(mean_val, color="tab:green", linestyle="--", label=f"mean={mean_val:.6g}")
            ax.axhline(
                median_val,
                color="tab:purple",
                linestyle="--",
                label=f"median={median_val:.6g}",
            )
            ax.set_title(f"{rel}\n{len(df):,} points; {span_seconds:,.0f} second span")
            ax.set_xlabel("Timestamp (UTC)")
            ax.set_ylabel("Sample")
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_png, dpi=300)
            fig.savefig(out_svg)
            plt.close(fig)

            # Save to CSV
            df.to_csv(out_csv, index=False)

        print()


if __name__ == "__main__":
    main()
