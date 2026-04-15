import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "No_Items_Added_InCart",
    "No_Checkout_Confirmed",
    "No_Checkout_Initiated",
    "No_Customer_Login",
    "No_Page_Viewed",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Session-like behavior (non-negative integer counts)
    no_page_viewed = rng.poisson(lam=7.0, size=n_rows)
    no_customer_login = rng.binomial(n=1, p=0.55, size=n_rows)  # 1 if logged in
    no_items_added = rng.poisson(lam=1.4, size=n_rows)
    no_checkout_initiated = rng.binomial(n=1, p=np.clip(no_items_added / 6.0, 0, 1), size=n_rows)
    no_checkout_confirmed = rng.binomial(
        n=1,
        p=np.clip(0.15 + 0.6 * no_checkout_initiated - 0.15 * (no_page_viewed < 2), 0, 1),
        size=n_rows,
    )

    # Make abandonment probability correlated with behavior
    # More pages and items reduce abandonment; initiating/confirming checkout strongly reduces abandonment.
    z = (
        1.1
        - 0.18 * np.log1p(no_page_viewed)
        - 0.22 * np.log1p(no_items_added)
        - 1.2 * no_checkout_initiated
        - 2.0 * no_checkout_confirmed
        - 0.25 * no_customer_login
        + rng.normal(0, 0.35, size=n_rows)
    )
    p_abandon = sigmoid(z)
    cart_abandoned = rng.binomial(n=1, p=p_abandon, size=n_rows)

    df = pd.DataFrame(
        {
            "No_Items_Added_InCart": no_items_added.astype(int),
            "No_Checkout_Confirmed": no_checkout_confirmed.astype(int),
            "No_Checkout_Initiated": no_checkout_initiated.astype(int),
            "No_Customer_Login": no_customer_login.astype(int),
            "No_Page_Viewed": no_page_viewed.astype(int),
            "Cart_Abandoned": cart_abandoned.astype(int),
        }
    )

    # Safety: ensure no negatives
    for c in FEATURES:
        df[c] = df[c].clip(lower=0)

    return df


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic cart-abandonment dataset.")
    ap.add_argument("--rows", type=int, default=1000, help="Number of rows to generate.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path. Default: uploads/synthetic_cart_abandonment_<timestamp>.csv",
    )
    args = ap.parse_args()

    df = make_synthetic(n_rows=max(1, args.rows), seed=args.seed)

    if args.out.strip():
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("uploads") / f"synthetic_cart_abandonment_{stamp}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print("Columns:", ", ".join(df.columns.tolist()))
    print("Shape:", df.shape)
    print("Cart_Abandoned balance:", df["Cart_Abandoned"].value_counts().to_dict())


if __name__ == "__main__":
    main()

