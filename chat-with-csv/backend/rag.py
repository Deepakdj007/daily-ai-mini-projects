# backend/rag.py

import pandas as pd
import io


def build_csv_context(file_bytes: bytes) -> str:
    """
    Transforms raw CSV bytes into a structured context string
    for injection into an LLM prompt.

    This is the "Retrieval" part of RAG. For CSV files we don't need
    embeddings or a vector database — we read the file, compute statistics,
    and format everything into a text block the model can reason over directly.

    Strategy:
    - Report total shape (rows x columns)
    - Show schema: column names, dtypes, null counts
    - Compute summary statistics for numeric columns
    - Show top 5 values for categorical columns
    - Include first 20 rows as a representative sample
    """
    df = pd.read_csv(io.BytesIO(file_bytes))

    lines = []

    # 1. Basic shape
    lines.append(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    # 2. Column schema
    lines.append("Columns (name | dtype | null count):")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        lines.append(f"  - {col!r}: {dtype}, {nulls} nulls")
    lines.append("")

    # 3. Numeric summary statistics
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        lines.append("Summary statistics (numeric columns):")
        stats = numeric_cols.describe().round(2)
        lines.append(stats.to_string())
        lines.append("")

    # 4. Categorical column top values (top 5 per column)
    categorical_cols = df.select_dtypes(include=["object", "category"])
    if not categorical_cols.empty:
        lines.append("Categorical columns (top 5 most frequent values each):")
        for col in categorical_cols.columns:
            counts = df[col].value_counts().head(5).to_dict()
            lines.append(f"  {col!r}: {counts}")
        lines.append("")

    # 5. Sample rows — gives the model a feel for the actual data format
    lines.append("Sample data (first 20 rows):")
    lines.append(df.head(20).to_string(index=False))

    return "\n".join(lines)