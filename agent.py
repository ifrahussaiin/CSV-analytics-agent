"""
Streamlit CSV/XLSX Dashboard Agent

Save this file as `dashboard_agent.py` and run:
    pip install -r requirements.txt
    streamlit run dashboard_agent.py

Requirements (put in requirements.txt):
    streamlit
    pandas
    plotly
    openpyxl

What it does:
- Loads a CSV or Excel file (from current directory or via file uploader)
- Performs simple automatic data cleaning (drop duplicates, trim strings, convert datetimes, fill missing values)
- Parses a natural-language prompt you enter to infer desired charts/colors (basic heuristics)
- Auto-selects sensible columns for: bar charts, line chart, pie charts
- Renders an interactive Streamlit dashboard with the requested charts
- Exposes controls in the sidebar to override automatic choices

Notes:
- The prompt parser is intentionally simple but robust for typical prompts (e.g. "use 2 bar charts, 1 line chart, 2 pie charts, green and blue colors").
- When auto-selection fails, use the sidebar selectors to pick columns.

"""

from typing import Optional, Tuple, List, Dict
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import re
import numpy as np
from datetime import datetime

st.set_page_config(page_title="CSV Dashboard Agent", layout="wide")

# ------------------------- Utilities -------------------------


def find_first_datafile() -> Optional[str]:
    # Search for common files in current directory
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    for pat in patterns:
        files = glob.glob(pat)
        files = [f for f in files if not f.startswith("~$")]
        if files:
            return files[0]
    return None


def load_data(path_or_buffer):
    if isinstance(path_or_buffer, str):
        if path_or_buffer.lower().endswith(".csv"):
            return pd.read_csv(path_or_buffer)
        else:
            return pd.read_excel(path_or_buffer)
    else:
        # file-like object from uploader
        filename = getattr(path_or_buffer, "name", "")
        if filename.lower().endswith(".csv"):
            return pd.read_csv(path_or_buffer)
        else:
            return pd.read_excel(path_or_buffer)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop fully empty columns and rows
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)

    # Trim strings and optionally unify casing for column names
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Try converting columns to datetime
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(5).tolist()
            joined = " ".join(sample)
            # crude heuristic: presence of digits and '-' or '/'
            if re.search(r"\d{4}-\d{2}-\d{2}", joined) or re.search(
                r"\d{1,2}/\d{1,2}/\d{2,4}", joined
            ):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

    # Fill missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                median = df[col].median()
                df[col].fillna(median, inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # do not fill datetimes
            continue
        else:
            if df[col].isna().any():
                try:
                    mode = df[col].mode().iloc[0]
                    df[col].fillna(mode, inplace=True)
                except Exception:
                    df[col].fillna("", inplace=True)

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def parse_prompt(prompt: str) -> Dict:
    """Simple parser to detect chart counts and colors from prompt."""
    p = prompt.lower()
    result = {"bars": 0, "lines": 0, "pies": 0, "colors": None}
    # numbers
    m = re.search(r"(\d+)\s*bar", p)
    if m:
        result["bars"] = int(m.group(1))
    else:
        # fuzzy: 'two bar charts'
        m = re.search(r"one|two|three|four|five|six", p)
        if m and "bar" in p:
            words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            result["bars"] = words.get(m.group(0), 0)

    m = re.search(r"(\d+)\s*line", p)
    if m:
        result["lines"] = int(m.group(1))
    elif "line chart" in p and result["lines"] == 0:
        # assume 1 if user mentions line chart
        result["lines"] = 1

    m = re.search(r"(\d+)\s*pie", p)
    if m:
        result["pies"] = int(m.group(1))
    else:
        if "pie chart" in p and result["pies"] == 0:
            result["pies"] = 1

    # colors
    colors = []
    if "green" in p:
        colors.append("#2ecc71")
    if "blue" in p:
        colors.append("#3498db")
    if "red" in p:
        colors.append("#e74c3c")
    if "orange" in p:
        colors.append("#f39c12")
    if not colors:
        colors = None
    result["colors"] = colors

    return result


def guess_columns_for_bar(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    # prefer categorical x and numeric y
    object_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if object_cols and numeric_cols:
        return object_cols[0], numeric_cols[0]
    elif object_cols:
        return object_cols[0], object_cols[0]
    elif len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    elif numeric_cols:
        return numeric_cols[0], numeric_cols[0]
    return None


def guess_columns_for_line(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    # prefer datetime x and numeric y
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if date_cols and numeric_cols:
        return date_cols[0], numeric_cols[0]
    elif len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    elif numeric_cols:
        return numeric_cols[0], numeric_cols[0]
    return None


def guess_columns_for_pie(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    # pie: categorical distribution; if numeric, pick top categories by grouping
    object_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if object_cols:
        return object_cols[0], None
    elif numeric_cols:
        return numeric_cols[0], None
    return None


# ------------------------- Streamlit App -------------------------

st.title("CSV/XLSX Dashboard Agent")
# Sidebar: file selection and prompt
with st.sidebar.expander("Data & Prompt"):
    use_current = st.checkbox(
        "Try to load a file from current directory (if exists)", value=True
    )
    detected = None
    if use_current:
        detected = find_first_datafile()
        st.write(
            "Detected file:" if detected else "No file detected in current directory"
        )
        if detected:
            st.write(detected)
    uploaded = st.file_uploader(
        "Or upload a CSV/XLSX file", type=["csv", "xlsx", "xls"]
    )
    prefer_file = st.text_input(
        "Or type filename in current directory (leave blank to use detected/uploaded)",
        value="",
    )
    st.markdown("---")
    prompt = st.text_area(
        "Dashboard prompt (natural language)",
        height=120,
        value="I have a csv in current directory and want a Streamlit dashboard. Use green and blue colours. 2 bar charts, 1 line chart, 2 pie charts. Please clean the data first.",
    )
    st.button("Generate Dashboard", key="generate")

# Determine file to load
chosen_path = None
if prefer_file:
    if os.path.exists(prefer_file):
        chosen_path = prefer_file
    else:
        st.sidebar.warning("File not found in current directory: " + prefer_file)

if uploaded is not None:
    df_raw = load_data(uploaded)
elif chosen_path:
    df_raw = load_data(chosen_path)
elif detected and use_current:
    try:
        df_raw = load_data(detected)
    except Exception as e:
        st.sidebar.error(f"Error loading detected file: {e}")
        df_raw = None
else:
    df_raw = None

if df_raw is None:
    st.info(
        "No data loaded yet. Upload a file or ensure a file exists in the current directory."
    )
    st.stop()

st.sidebar.success("Data loaded: %d rows × %d columns" % df_raw.shape)

# Cleaning
with st.expander("Preview & Clean Data (auto)"):
    st.subheader("Raw data preview")
    st.dataframe(df_raw.head(50))
    if st.button("Auto-clean data"):
        df = clean_dataframe(df_raw)
        st.session_state["df"] = df
        st.success('Auto-cleaning applied. Check "Cleaned data" preview below.')
    else:
        if "df" not in st.session_state:
            # default to cleaned
            st.session_state["df"] = clean_dataframe(df_raw)
    st.subheader("Cleaned data preview")
    st.dataframe(st.session_state["df"].head(50))

# Parse prompt
parsed = parse_prompt(prompt)

st.sidebar.markdown("**Auto-parsed from prompt**")
st.sidebar.write(parsed)

# Determine colors
color_seq = parsed["colors"] or ["#2ECC71", "#3498DB"]
# If only one color given, duplicate to ensure variety
if isinstance(color_seq, list) and len(color_seq) == 1:
    color_seq = color_seq * 2

# Number of charts
bars_n = parsed["bars"] if parsed["bars"] else 0
lines_n = parsed["lines"] if parsed["lines"] else 0
pies_n = parsed["pies"] if parsed["pies"] else 0

# Allow override
st.sidebar.markdown("---")
bars_n = st.sidebar.number_input(
    "Number of bar charts", min_value=0, max_value=6, value=bars_n
)
lines_n = st.sidebar.number_input(
    "Number of line charts", min_value=0, max_value=6, value=lines_n
)
pies_n = st.sidebar.number_input(
    "Number of pie charts", min_value=0, max_value=6, value=pies_n
)

# Prepare space for charts
df = st.session_state["df"]

# Helper to render one bar chart


def render_bar(
    df: pd.DataFrame, x_col: str, y_col: Optional[str], title: str, colors: List[str]
):
    if y_col is None or x_col == y_col:
        # treat as value counts
        try:
            series = df[x_col].astype(str).value_counts().nlargest(20)
            fig = px.bar(
                series.reset_index(),
                x="index",
                y=x_col,
                labels={"index": x_col},
                title=title,
            )
            fig.update_traces(marker_color=colors[0 : len(series)])
        except Exception as e:
            st.error("Could not render bar chart: " + str(e))
            return
    else:
        # aggregate
        try:
            agg = (
                df.groupby(x_col)[y_col]
                .sum()
                .reset_index()
                .sort_values(y_col, ascending=False)
                .head(20)
            )
            fig = px.bar(agg, x=x_col, y=y_col, title=title)
            fig.update_traces(marker_color=colors[0 : len(agg)])
        except Exception as e:
            st.error("Could not render bar chart: " + str(e))
            return
    st.plotly_chart(fig, use_container_width=True)


def render_line(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, colors: List[str]
):
    try:
        fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Could not render line chart: " + str(e))


def render_pie(df: pd.DataFrame, col: str, title: str, colors: List[str]):
    try:
        series = df[col].astype(str).value_counts().reset_index().head(10)
        series.columns = [col, "count"]
        fig = px.pie(series, names=col, values="count", title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Could not render pie chart: " + str(e))


# Auto-guess columns for each chart

bar_choices = []
for i in range(bars_n):
    guessed = guess_columns_for_bar(df)
    if guessed:
        default_x, default_y = guessed
    else:
        default_x, default_y = (df.columns[0], df.columns[0])
    st.sidebar.markdown(f"**Bar chart {i+1}**")
    x = st.sidebar.selectbox(
        f"Bar {i+1} - x (category)",
        options=list(df.columns),
        index=list(df.columns).index(default_x) if default_x in df.columns else 0,
        key=f"bar_x_{i}",
    )
    y = st.sidebar.selectbox(
        f"Bar {i+1} - y (numeric or leave same for counts)",
        options=list(df.columns),
        index=list(df.columns).index(default_y) if default_y in df.columns else 0,
        key=f"bar_y_{i}",
    )
    bar_choices.append((x, y))

line_choices = []
for i in range(lines_n):
    guessed = guess_columns_for_line(df)
    if guessed:
        default_x, default_y = guessed
    else:
        default_x, default_y = (
            df.columns[0],
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
        )
    st.sidebar.markdown(f"**Line chart {i+1}**")
    x = st.sidebar.selectbox(
        f"Line {i+1} - x (time or numeric)",
        options=list(df.columns),
        index=list(df.columns).index(default_x) if default_x in df.columns else 0,
        key=f"line_x_{i}",
    )
    y = st.sidebar.selectbox(
        f"Line {i+1} - y (numeric)",
        options=list(df.columns),
        index=list(df.columns).index(default_y) if default_y in df.columns else 0,
        key=f"line_y_{i}",
    )
    line_choices.append((x, y))

pie_choices = []
for i in range(pies_n):
    guessed = guess_columns_for_pie(df)
    if guessed:
        default_col = guessed[0]
    else:
        default_col = df.columns[0]
    st.sidebar.markdown(f"**Pie chart {i+1}**")
    col = st.sidebar.selectbox(
        f"Pie {i+1} - column",
        options=list(df.columns),
        index=list(df.columns).index(default_col) if default_col in df.columns else 0,
        key=f"pie_col_{i}",
    )
    pie_choices.append(col)

# Layout: put charts into columns row by row

chart_idx = 0
cols_per_row = 2

# Bars first
if bars_n > 0:
    st.header("Bar charts")
    for i, (x, y) in enumerate(bar_choices):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        with row[i % cols_per_row]:
            title = (
                f"Bar chart {i+1}: {y} by {x}"
                if x != y
                else f"Bar chart {i+1}: Counts of {x}"
            )
            render_bar(df, x, y if x != y else None, title, color_seq)

# Lines next
if lines_n > 0:
    st.header("Line charts")
    for i, (x, y) in enumerate(line_choices):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        with row[i % cols_per_row]:
            title = f"Line chart {i+1}: {y} over {x}"
            render_line(df, x, y, title, color_seq)

# Pies
if pies_n > 0:
    st.header("Pie charts")
    for i, col in enumerate(pie_choices):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        with row[i % cols_per_row]:
            title = f"Pie chart {i+1}: {col} distribution"
            render_pie(df, col, title, color_seq)

st.sidebar.markdown("---")
st.sidebar.write(
    "Tip: if automatic picks are wrong, change the selectors in the sidebar. You can also edit the prompt and re-generate."
)
