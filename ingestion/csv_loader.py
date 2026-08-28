import pandas as pd


class CSVLoader:
    def extract(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            rows.append(row_text)
        return "\n".join(rows)
