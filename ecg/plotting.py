import altair as alt
import polars as pl


def plot_reconstruction(input_df: pl.DataFrame, output_df: pl.DataFrame, target: str, pred: str) -> alt.Chart:
    """Plot a single example and it's reconstruction."""
    input_plot_df = pl.concat(
        [
            input_df.select(col)
            .rename({col: "signal"})
            .with_columns(
                pl.Series("case", [i] * input_df.shape[0]),
                pl.Series("measurement", list(range(input_df.shape[0]))),
                pl.Series("class", ["original"] * input_df.shape[0]),
            )
            for i, col in enumerate(input_df.columns)
        ],
        how="vertical",
    ).with_row_index()

    output_plot_df = pl.concat(
        [
            output_df.select(col)
            .rename({col: "signal"})
            .with_columns(
                pl.Series("case", [i] * output_df.shape[0]),
                pl.Series("measurement", list(range(output_df.shape[0]))),
                pl.Series("class", ["reconstructed"] * output_df.shape[0]),
            )
            for i, col in enumerate(output_df.columns)
        ],
        how="vertical",
    ).with_row_index()

    plot_df = pl.concat([input_plot_df, output_plot_df])

    return (
        alt.Chart(plot_df, title=f"Reconstruction ({target} | {pred})")
        .mark_line()
        .encode(
            x=alt.X("measurement", title="measurement"),
            y=alt.Y("signal", title="signal", scale=alt.Scale(domain=[0.0, 1.2])),
            color=alt.Color("class:N", title=None),
            detail="case",
        )
        .properties(height=300, width=600)
    )
