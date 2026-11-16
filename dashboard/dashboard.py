import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="E-Commerce Data Dashboard", page_icon="📊", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
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
st.markdown(f"**Wilayah Terpilih:** `{selected_state}` | **Periode:** `{main_df['order_purchase_timestamp'].min().strftime('%d %b %Y')}` – `{main_df['order_purchase_timestamp'].max().strftime('%d %b %Y')}`")

# Data level order unik agar metrik order & review tidak terhitung ganda
orders_unique = main_df.drop_duplicates(subset=["order_id"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_orders = orders_unique["order_id"].nunique()
    st.metric("Total Pesanan", f"{total_orders:,}")
with col2:
    total_revenue = main_df["price"].sum()
    st.metric("Total Pendapatan Produk", f"R$ {total_revenue:,.2f}")
with col3:
    avg_review = orders_unique["review_score"].mean()
    st.metric("Rata-Rata Skor Ulasan", f"{avg_review:.2f} / 5.0" if pd.notna(avg_review) else "N/A")
with col4:
    late_rate = (orders_unique["is_late"].mean() * 100) if len(orders_unique) > 0 else 0
    st.metric("Persentase Keterlambatan", f"{late_rate:.2f}%")

st.divider()

# ================= SECTION 1: KETERLAMBATAN & KEPUASAN PELANGGAN =================
st.subheader("1. Performa Pengiriman & Dampaknya terhadap Skor Ulasan")

col_left, col_right = st.columns([1.4, 1])

with col_left:
    orders_unique = orders_unique.copy()
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
    ax.set_title("Rata-Rata Skor Ulasan Berdasarkan Ketepatan Waktu", fontsize=12, fontweight="bold")
    ax.set_xlabel("Status Pengiriman")
    ax.set_ylabel("Skor Ulasan (1 - 5)")
    ax.set_ylim(0, 5.2)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontweight="bold", xytext=(0, 4), textcoords="offset points")
    sns.despine(ax=ax)
    st.pyplot(fig)

st.divider()

# ================= SECTION 2: TOP KATEGORI PRODUK =================
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
ax.set_title(f"Top 5 Kategori Produk Terlaris ({selected_state})", fontsize=12, fontweight="bold")
ax.set_xlabel("Total Pendapatan (BRL)")
ax.set_ylabel("Kategori Produk")
if len(top_categories) > 0:
    ax.set_xlim(0, top_categories["price"].max() * 1.18)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"R$ {width:,.2f}", (width, p.get_y() + p.get_height() / 2.),
                ha="left", va="center", fontweight="bold", xytext=(6, 0), textcoords="offset points")
sns.despine(ax=ax)
st.pyplot(fig)

st.divider()

# ================= SECTION 3: ANALISIS LANJUTAN (RFM & SEGMENTASI) =================
st.subheader("3. Analisis Lanjutan: RFM & Segmentasi Pelanggan")

recent_date = orders_unique["order_purchase_timestamp"].max()
rfm_df = orders_unique.groupby("customer_unique_id", as_index=False).agg(
    recency=("order_purchase_timestamp", lambda x: (recent_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
)

rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
with rfm_col1:
    st.metric("Rata-Rata Recency (Hari)", f"{rfm_df['recency'].mean():.1f} Hari")
with rfm_col2:
    st.metric("Rata-Rata Frequency", f"{rfm_df['frequency'].mean():.2f} Transaksi")
with rfm_col3:
    st.metric("Rata-Rata Monetary", f"R$ {rfm_df['monetary'].mean():,.2f}")

# Binning / Manual Grouping
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
