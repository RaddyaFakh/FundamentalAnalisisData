import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr, spearmanr

# Konfigurasi Halaman
st.set_page_config(page_title="E-Commerce Data Dashboard", page_icon="📊", layout="wide")
sns.set_style("whitegrid")

# Load Data
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "main_data.csv")
    df = pd.read_csv(file_path)
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    df["delivery_diff_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    return df

all_df = load_data()

# ================= SIDEBAR FILTER =================
with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.markdown("Filter data untuk menyesuaikan tampilan analisis pada dashboard.")

    min_date = all_df["order_purchase_timestamp"].min().date()
    max_date = all_df["order_purchase_timestamp"].max().date()

    date_range = st.date_input(
        label="Rentang Tanggal Pembelian",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    state_list = ["All States"] + sorted(all_df["customer_state"].dropna().unique().tolist())
    selected_state = st.selectbox("Pilih Negara Bagian (State)", options=state_list, index=0)

# Terapkan Filter
if len(date_range) == 2:
    start_date, end_date = date_range
    main_df = all_df[
        (all_df["order_purchase_timestamp"].dt.date >= start_date) &
        (all_df["order_purchase_timestamp"].dt.date <= end_date)
    ]
else:
    main_df = all_df.copy()

if selected_state != "All States":
    main_df = main_df[main_df["customer_state"] == selected_state]

# ================= HEADER & KPI METRICS =================
st.title("📊 E-Commerce Public Dataset Dashboard")
if len(main_df) > 0:
    st.markdown(
        f"**Wilayah Terpilih:** `{selected_state}` | "
        f"**Periode:** `{main_df['order_purchase_timestamp'].min().strftime('%d %b %Y')}` – "
        f"`{main_df['order_purchase_timestamp'].max().strftime('%d %b %Y')}`"
    )
else:
    st.warning("Tidak ada data transaksi pada rentang filter yang dipilih.")
    st.stop()

orders_unique = main_df.drop_duplicates(subset=["order_id"]).copy()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Pesanan", f"{orders_unique['order_id'].nunique():,}")
with col2:
    st.metric("Total Pendapatan Produk", f"R$ {main_df['price'].sum():,.2f}")
with col3:
    avg_review = orders_unique["review_score"].mean()
    st.metric("Rata-Rata Skor Ulasan", f"{avg_review:.2f} / 5.0" if pd.notna(avg_review) else "N/A")
with col4:
    late_rate = orders_unique["is_late"].mean() * 100
    st.metric("Persentase Keterlambatan", f"{late_rate:.2f}%")

st.divider()

# ================= SECTION 1: PERTANYAAN 1 =================
st.subheader("1. Tren Keterlambatan Pengiriman & Korelasi dengan Skor Ulasan")

# Hitung Korelasi Pearson & Spearman
corr_df = orders_unique.dropna(subset=["delivery_diff_days", "review_score"])
if len(corr_df) > 2:
    r_pearson, p_pearson = pearsonr(corr_df["delivery_diff_days"], corr_df["review_score"])
    r_spearman, p_spearman = spearmanr(corr_df["delivery_diff_days"], corr_df["review_score"])

    k_col1, k_col2, k_col3 = st.columns(3)
    with k_col1:
        st.metric("Korelasi Pearson (r)", f"{r_pearson:.4f}")
    with k_col2:
        st.metric("Korelasi Spearman (ρ)", f"{r_spearman:.4f}")
    with k_col3:
        st.metric("Signifikansi (p-value)", f"{p_pearson:.2e}")

col_left, col_right = st.columns([1.4, 1])

with col_left:
    orders_unique["year_month"] = orders_unique["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly_delay = orders_unique.groupby("year_month", as_index=False).agg(
        late_percentage=("is_late", lambda x: round(x.mean() * 100, 2))
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=monthly_delay, x="year_month", y="late_percentage", marker="o", linewidth=2.5, color="#d62728", ax=ax)
    ax.set_title("Tren Persentase Keterlambatan Pengiriman per Bulan", fontsize=12, fontweight="bold")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Keterlambatan (%)")
    plt.xticks(rotation=45)
    sns.despine(ax=ax)
    st.pyplot(fig)

with col_right:
    delay_review = orders_unique.groupby("is_late", as_index=False)["review_score"].mean()
    delay_review["Status"] = delay_review["is_late"].map({False: "Tepat Waktu", True: "Terlambat"})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(
        data=delay_review,
        x="Status",
        y="review_score",
        palette=["#2b5c8f", "#d62728"],
        hue="Status",
        legend=False,
        width=0.5,
        ax=ax
    )
    ax.set_title("Rata-Rata Skor Ulasan: Tepat Waktu vs Terlambat", fontsize=12, fontweight="bold")
    ax.set_xlabel("Status Pengiriman")
    ax.set_ylabel("Skor Ulasan (1 - 5)")
    ax.set_ylim(0, 5.2)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom", fontweight="bold", xytext=(0, 4), textcoords="offset points"
        )
    sns.despine(ax=ax)
    st.pyplot(fig)

# Scatter Plot Hubungan Selisih Hari Pengiriman vs Skor Ulasan
if len(corr_df) > 2:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    np.random.seed(42)
    y_jitter = corr_df["review_score"] + np.random.uniform(-0.22, 0.22, size=len(corr_df))

    ax.scatter(
        corr_df["delivery_diff_days"],
        y_jitter,
        alpha=0.15,
        s=12,
        color="#2b5c8f",
        label="Data Pesanan"
    )
    sns.regplot(
        data=corr_df,
        x="delivery_diff_days",
        y="review_score",
        scatter=False,
        color="#d62728",
        line_kws={"linewidth": 2.5, "label": "Garis Tren Regresi"},
        ax=ax
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, label="Batas Estimasi (0 Hari)")

    stats_box = f"Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})\nSpearman ρ = {r_spearman:.4f} (p = {p_spearman:.2e})"
    ax.text(
        0.98, 0.92, stats_box,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc")
    )

    ax.set_title("Scatter Plot: Hubungan Selisih Hari Pengiriman dengan Skor Ulasan", fontsize=12, fontweight="bold")
    ax.set_xlabel("Selisih Hari dari Estimasi (Negatif = Lebih Cepat, Positif = Terlambat)")
    ax.set_ylabel("Skor Ulasan (1 - 5)")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc="lower left")
    sns.despine(ax=ax)
    st.pyplot(fig)

st.divider()

# ================= SECTION 2: PERTANYAAN 2 =================
st.subheader("2. Top 5 Kategori Produk Berdasarkan Total Pendapatan")

top_categories = (
    main_df.groupby("product_category_name_english", as_index=False)["price"]
    .sum()
    .sort_values(by="price", ascending=False)
    .head(5)
)

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = ["#1f77b4"] + ["#aec7e8"] * (len(top_categories) - 1)
sns.barplot(
    data=top_categories,
    x="price",
    y="product_category_name_english",
    palette=colors,
    hue="product_category_name_english",
    legend=False,
    ax=ax
)
ax.set_title(f"Top 5 Kategori Produk Berdasarkan Total Pendapatan ({selected_state})", fontsize=12, fontweight="bold")
ax.set_xlabel("Total Pendapatan (BRL)")
ax.set_ylabel("Kategori Produk")
if len(top_categories) > 0:
    ax.set_xlim(0, top_categories["price"].max() * 1.18)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(
        f"R$ {width:,.2f}",
        (width, p.get_y() + p.get_height() / 2.),
        ha="left", va="center", fontweight="bold", xytext=(6, 0), textcoords="offset points"
    )
sns.despine(ax=ax)
st.pyplot(fig)

st.divider()

# ================= SECTION 3: ANALISIS LANJUTAN (RFM) =================
st.subheader("3. Analisis Lanjutan: RFM & Segmentasi Pelanggan")

recent_date = orders_unique["order_purchase_timestamp"].max()
rfm_df = orders_unique.groupby("customer_unique_id", as_index=False).agg(
    recency=("order_purchase_timestamp", lambda x: (recent_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
)

rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
with rfm_col1:
    st.metric("Rata-Rata Recency", f"{rfm_df['recency'].mean():.1f} Hari")
with rfm_col2:
    st.metric("Rata-Rata Frequency", f"{rfm_df['frequency'].mean():.2f} Transaksi")
with rfm_col3:
    st.metric("Rata-Rata Monetary", f"R$ {rfm_df['monetary'].mean():,.2f}")

rfm_df["recency_segment"] = pd.cut(
    rfm_df["recency"],
    bins=[-1, 90, 180, max(181, rfm_df["recency"].max())],
    labels=["Active (<= 90 Hari)", "Warm (91-180 Hari)", "Inactive (> 180 Hari)"]
)
rfm_df["monetary_segment"] = pd.cut(
    rfm_df["monetary"],
    bins=[-1, 100, 300, max(301, rfm_df["monetary"].max())],
    labels=["Low Spender", "Medium Spender", "High Spender"]
)

seg_col1, seg_col2 = st.columns(2)
with seg_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=rfm_df, x="recency_segment", palette="Blues_r", hue="recency_segment", legend=False, ax=ax)
    ax.set_title("Distribusi Segmen Recency Pelanggan", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah Pelanggan")
    sns.despine(ax=ax)
    st.pyplot(fig)

with seg_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=rfm_df, x="monetary_segment", palette="Greens", hue="monetary_segment", legend=False, ax=ax)
    ax.set_title("Distribusi Segmen Monetary Pelanggan", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah Pelanggan")
    sns.despine(ax=ax)
    st.pyplot(fig)

st.caption("Proyek Analisis Data: E-Commerce Public Dataset — Muhammad Raddya Fakhreza")
