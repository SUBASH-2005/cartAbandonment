# recommender.py — No-image content-based recommender (TF-IDF)

from __future__ import annotations
from typing import List, Optional
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLS = ["id", "title", "category", "description", "price"]

class ProductRecommender:
    def __init__(self, products_csv: str):
        if not os.path.exists(products_csv):
            raise FileNotFoundError(f"products_csv not found: {products_csv}")
        self.products_csv = products_csv
        self.df = pd.read_csv(products_csv)
        self._normalize_schema()

        self.df["_blob"] = (
            self.df["title"].fillna("").astype(str) + " " +
            self.df["category"].fillna("").astype(str) + " " +
            self.df["description"].fillna("").astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.tfidf = self.vectorizer.fit_transform(self.df["_blob"].tolist())

        self.id_to_idx = {int(r["id"]): i for i, r in self.df.reset_index().iterrows()}

    def _normalize_schema(self):
        cols = {c.lower(): c for c in self.df.columns}
        alias = {"product_id": "id", "sku": "id", "name": "title", "product_name": "title", "details": "description"}
        for k, v in alias.items():
            if k in cols and v not in self.df.columns:
                self.df.rename(columns={cols[k]: v}, inplace=True)

        miss = [c for c in REQUIRED_COLS if c not in self.df.columns]
        if miss:
            raise ValueError(f"products.csv missing columns: {miss}. Expected: {REQUIRED_COLS}")

        self.df["id"] = pd.to_numeric(self.df["id"], errors="coerce").astype("Int64")
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")

    def recommend(self, product_id: int, k: int = 6) -> pd.DataFrame:
        if product_id not in self.id_to_idx:
            raise KeyError(f"Unknown product_id: {product_id}")
        idx = self.id_to_idx[product_id]
        row = self.tfidf[idx]
        sims = cosine_similarity(row, self.tfidf).ravel()
        sims[idx] = -1.0
        top_idx = np.argsort(-sims)[:k]
        out = self.df.iloc[top_idx].copy()
        out["score"] = sims[top_idx]
        return out[["id", "title", "category", "price", "description", "score"]].reset_index(drop=True)

    def get_all_titles(self) -> List[str]:
        return self.df["title"].astype(str).tolist()

    def id_for_title(self, title: str) -> Optional[int]:
        row = self.df.loc[self.df["title"] == title]
        if row.empty:
            return None
        return int(row.iloc[0]["id"])
