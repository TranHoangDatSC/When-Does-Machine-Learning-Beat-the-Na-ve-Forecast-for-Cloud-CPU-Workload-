"""Sinh năm hình của bài báo từ bảng kết quả.

    python scripts/hinh_bai_bao.py
    python scripts/hinh_bai_bao.py --goc results --out hinh

Mặc định đọc `ket_qua_tham_chieu/`, nên chạy được ngay sau khi clone mà không cần dữ liệu
thô và không cần chạy lại thí nghiệm. Truyền `--goc results` để vẽ từ lượt chạy của bạn.

Quy ước trình bày, theo yêu cầu của tạp chí
-------------------------------------------

Tạp chí chỉ nhận ảnh màu khi có thoả thuận riêng, và yêu cầu 600 dpi cho hình nét. Vì vậy
mọi hình ở đây là **đen trắng**: các đường phân biệt nhau bằng kiểu nét và ký hiệu chứ
không bằng màu, các ô của hình nhiệt tô bằng một thang xám duy nhất và in kèm số. Hình
đọc được cả khi in đen trắng lẫn khi người đọc bị mù màu.

Năm hình
--------

    hinh1_quy_trinh       sơ đồ quy trình, từ dữ liệu thô tới kiểm định
    hinh2_phan_bo_cpu     phân phối tải CPU của ba môi trường, giải thích vì sao hai
                          tiêu chí đánh giá bất đồng ở Bitbrains mà không ở Alibaba
    hinh3_ban_do_ket_luan hình nhiệt số mô hình vượt naive, theo tiêu chí và chuẩn hoá
    hinh4_on_dinh         tỉ số theo từng giai đoạn thời gian, kèm kết luận của kiểm định
    hinh5_chuyen_giao     hai tỉ số của mọi chiều chuyển giao, kèm bốn góc phần tư
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DPI = 600

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Chữ trong hình phải đọc được sau khi tạp chí thu nhỏ hình, nên đặt cỡ tuyệt đối.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.3,
    "grid.color": "0.85",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MOI_TRUONG = {"E1": "E1 Bitbrains fastStorage",
              "E2": "E2 Bitbrains Rnd",
              "E3": "E3 Alibaba"}
MAU_MT = {"E1": "#0072B2", "E2": "#D55E00", "E3": "#009E73"}
# Màu theo MÔ HÌNH, cho hình so nhiều mô hình trên cùng một ô.
MAU_MH = {"xgb_mae": "#0072B2", "xgb": "#D55E00",
          "ridge": "#009E73", "ma6": "#56B4E9"}
NET = {"E1": "-", "E2": "--", "E3": ":"}
KY_HIEU = {"E1": "o", "E2": "s", "E3": "^"}


def _luu(fig, out: Path, ten: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for duoi in ("png", "pdf"):
        fig.savefig(out / f"{ten}.{duoi}", dpi=DPI)
    plt.close(fig)
    print(f"  {ten}.png, {ten}.pdf")


# ------------------------------------------------------------------ Hình 1
def hinh1(out: Path) -> None:
    """Sơ đồ quy trình. Vẽ tay bằng hộp và mũi tên để không phụ thuộc công cụ ngoài."""
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    hop = [
        (0.2, 3.4, 2.1, 1.0, "Raw traces\nE1, E2, E3"),
        (2.7, 3.4, 2.1, 1.0, "Resample to\n5-minute grid"),
        (5.2, 3.4, 2.1, 1.0, "Filter on\ntraining part"),
        (7.7, 3.4, 2.1, 1.0, "Feature matrix\n20 features"),
        (7.7, 1.8, 2.1, 1.0, "Split\ntrain / val / test"),
        (5.2, 1.8, 2.1, 1.0, "Select hyper-\nparameters on val"),
        (2.7, 1.8, 2.1, 1.0, "Refit on\ntrain + val"),
        (0.2, 1.8, 2.1, 1.0, "Score test\nonce"),
        (0.2, 0.2, 4.6, 1.0, "Intersect to common evaluation rows"),
        (5.2, 0.2, 4.6, 1.0, "Metrics, paired tests, Holm correction"),
    ]
    for x, y, w, h, nhan in hop:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.7, ec="black"))
        ax.text(x + w / 2, y + h / 2, nhan, ha="center", va="center", fontsize=6.5)

    def mui_ten(a, b, cong=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color="black",
                                     connectionstyle=f"arc3,rad={cong}"))

    for i in range(3):
        mui_ten((2.3 + 2.5 * i, 3.9), (2.7 + 2.5 * i, 3.9))
    mui_ten((8.75, 3.4), (8.75, 2.8))
    for i in range(3):
        mui_ten((7.7 - 2.5 * i, 2.3), (7.3 - 2.5 * i, 2.3))
    mui_ten((1.25, 1.8), (1.25, 1.2))                   
    mui_ten((4.8, 0.7), (5.2, 0.7))
    _luu(fig, out, "hinh1_quy_trinh")


# ------------------------------------------------------------------ Hình 2
def hinh2(goc: Path, out: Path) -> None:
    """Phân phối tải CPU, một biểu đồ cột cho mỗi môi trường.

    Hình đỡ cho lập luận vì sao hai tiêu chí đánh giá bất đồng ở Bitbrains mà không ở
    Alibaba, nên thứ cần thấy là khối lượng quan sát dồn ở đâu trên thang 0 đến 100.

    Màu chỉ lặp lại điều tiêu đề từng ô đã nói nên hình vẫn đọc được khi in đen trắng.
    Bộ ba màu đã qua sáu phép kiểm của bộ kiểm bảng màu, gồm cả tách biệt khi mù màu:
    cặp gần nhau nhất cách ΔE 11,0 ở deutan và 25,8 ở thị lực thường.
    """
    p = goc / "phan_bo_cpu_histogram.csv"
    if not p.exists():
        print(f"  bỏ qua hình 2: thiếu {p.name}")
        return
    d = pd.read_csv(p)

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.8), sharey=True)
    for ax, e in zip(axes, ("E1", "E2", "E3")):
        x = d[d.env == e]
        ax.bar(x.canh_duoi, x.pct_diem, width=1.0, align="edge",
               color=MAU_MT[e], edgecolor="none")
        tv = float(x.trung_vi.iloc[0])
        ax.axvline(tv, color="black", ls="--", lw=0.8)
        ax.annotate(f"median {tv:.2f}%", (tv, 0.97),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(3, -1), fontsize=6.5, va="top")
        ax.set_title(MOI_TRUONG[e], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("CPU (%)")
        ax.grid(True, axis="y", lw=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of observations (%)")
    _luu(fig, out, "hinh2_phan_bo_cpu")


# ------------------------------------------------------------------ Hình 3
def hinh3(goc: Path, out: Path) -> None:
    """Hình nhiệt: số mô hình vượt naive, theo tiêu chí và chế độ chuẩn hoá."""
    k = pd.read_csv(goc / "tong_hop" / "kiem_dinh.csv")
    a = k[k.thi_nghiem == "tn_a"]
    cot, nhan = [], []
    for m, tm in (("mae", "MAE"), ("rmse", "RMSE"), ("mase", "MASE")):
        for mode in ("N0", "N1"):
            cot.append((m, mode)); nhan.append(f"{tm}\n{mode}")
    hang = [(e, h) for e in ("E1", "E2", "E3") for h in (1, 6, 12)]

    dat = np.zeros((len(hang), len(cot)))
    chu = np.empty((len(hang), len(cot)), dtype=object)
    for i, (e, h) in enumerate(hang):
        for j, (m, mode) in enumerate(cot):
            x = a[(a.dich == e) & (a.h == h) & (a["mode"] == mode)]
            thang, tong = int(x[f"{m}_vuot_naive"].sum()), len(x)
            dat[i, j] = thang / tong if tong else np.nan
            chu[i, j] = f"{thang}/{tong}"

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    im = ax.imshow(dat, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cot))); ax.set_xticklabels(nhan)
    ax.set_yticks(range(len(hang)))
    ax.set_yticklabels([f"{e}  h={h}" for e, h in hang])
    for i in range(len(hang)):
        for j in range(len(cot)):
            val = dat[i, j]
            chu_color = "white" if (val < 0.3 or val > 0.7) else "black"
            ax.text(j, i, chu[i, j], ha="center", va="center", fontsize=6,
                    color=chu_color)
    for i in (2.5, 5.5):
        ax.axhline(i, color="black", lw=0.6)
    for j in (1.5, 3.5):
        ax.axvline(j, color="black", lw=0.6)
    ax.set_xticks(np.arange(-0.5, len(cot), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hang), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Fraction of models beating the naive forecast", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    _luu(fig, out, "hinh3_ban_do_ket_luan")


# ------------------------------------------------------------------ Hình 4
def hinh4(goc: Path, out: Path) -> None:
    """Hai tỉ số của mọi chiều chuyển giao. Góc phần tư nói ngay kết luận."""
    c = pd.read_csv(goc / "tong_hop" / "chuyen_giao.csv")
    c = c[c.thi_nghiem == "tn_c"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    x_lo, x_hi = float(c.L.min()) / 1.25, float(c.L.max()) * 1.25
    y_lo = float(c.so_voi_naive_tai_dich.min()) / 1.25
    y_hi = float(c.so_voi_naive_tai_dich.max()) * 1.25 * 3.2
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    ax.add_patch(Rectangle((x_lo, 1), 1 - x_lo, y_hi - 1, transform=ax.transData,
                       facecolor="#f0f0f0", edgecolor="none", zorder=0))
    for e in ("E1", "E2", "E3"):
        x = c[c.nguon == e]
        ax.scatter(x.L, x.so_voi_naive_tai_dich, s=16, marker=KY_HIEU[e],
                   facecolors="none", edgecolors=MAU_MT[e], linewidths=0.9,
                   label=f"trained on {e}", zorder=3)
    ax.axvline(1, color="black", lw=0.6, zorder=2)
    ax.axhline(1, color="black", lw=0.6, zorder=2)
    ax.set_xlabel("Ratio to the model trained at the target")
    ax.set_ylabel("Ratio to the naive forecast at the target")
    ax.grid(True, which="major", lw=0.3, zorder=1)

    n_ok = int(((c.L < 1) & (c.so_voi_naive_tai_dich < 1)).sum())
    n_bay = int(((c.L < 1) & (c.so_voi_naive_tai_dich >= 1)).sum())
    ax.text(0.20, 0.985,
            f"Reuse works, both ratios below 1: {n_ok} of {len(c)}\n"
            f"Shaded: beats the target model yet loses to the\n"
            f"target naive forecast, {n_bay} settings",
            transform=ax.transAxes, fontsize=6, va="top", ha="left")
    ax.legend(loc="lower right")
    _luu(fig, out, "hinh5_chuyen_giao")


# ------------------------------------------------------------------ Hình 5
def hinh5(goc: Path, out: Path) -> None:
    """Tỉ số so với naive qua ba giai đoạn thời gian, cho bốn mô hình được huấn luyện."""
    k = pd.read_csv(goc / "tong_hop" / "kiem_dinh.csv")
    b = k[(k.thi_nghiem == "tn_b") & (k["mode"] == "N1") & (k.h == 1)]
    if b.empty:
        print("  bỏ qua hình 5: chưa có kết quả của thí nghiệm theo thời gian")
        return
    envs = ["E1L", "E2m07", "E2m09"]
    mo_hinh = ["xgb_mae", "xgb", "ridge", "ma6"]
    nhan_mh = {"xgb_mae": "tree, absolute loss", "xgb": "tree, squared loss",
               "ridge": "ridge", "ma6": "moving average"}
    net_mh = {"xgb_mae": "-", "xgb": "--", "ridge": "-.", "ma6": ":"}
    ky_mh = {"xgb_mae": "o", "xgb": "s", "ridge": "^", "ma6": "v"}
    color_mh = MAU_MH
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.1), sharey=True)
    folds = sorted(b.fold.unique())
    for ax, e in zip(axes, envs):
        for m in mo_hinh:
            x = b[(b.dich == e) & (b.model == m)].set_index("fold")
            y = [x.mae_ti_so_p50.get(f, np.nan) for f in folds]
            ax.plot(range(len(folds)), y, net_mh[m], color=color_mh[m], lw=1.2,
                    label=nhan_mh[m], zorder=2)
            for i, f in enumerate(folds):
                if f not in x.index:
                    continue
                thang = bool(x.mae_vuot_naive.get(f, False))
                ax.plot(i, x.mae_ti_so_p50.get(f), ky_mh[m], ms=4.2, 
                        color=color_mh[m], 
                        mfc=color_mh[m] if thang else "white",
                        mew=0.8, zorder=3)                
        ax.axhline(1, color="gray", ls="--", lw=0.7, zorder=1)
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([f"W{i + 1}" for i in range(len(folds))])
        ax.set_xlim(-0.35, len(folds) - 0.65)
        ax.set_title(e, fontsize=7, fontweight="bold")
        ax.grid(True, axis="y", lw=0.3, ls=":")
        
    axes[0].set_ylabel("Ratio to the naive forecast", fontsize=6.5)
    axes[1].set_xlabel("Retraining window", fontsize=6.5)
    
    han, nhan = axes[0].get_legend_handles_labels()
    fig.legend(han, nhan, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22), fontsize=6)
    
    fig.text(0.5, -0.32, "Filled marker: beats the naive forecast after Holm "
                         "correction. Open marker: does not.",
             ha="center", fontsize=5.5, style="italic")
             
    _luu(fig, out, "hinh4_on_dinh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--goc", default="ket_qua_tham_chieu")
    ap.add_argument("--out", default="hinh")
    a = ap.parse_args()
    goc, out = ROOT / a.goc, ROOT / a.out
    if not goc.exists():
        print(f"Thiếu {goc}.")
        return 2
    print(f"Vẽ từ {a.goc} vào {a.out}/ ({DPI} dpi, đen trắng)")
    hinh1(out)
    hinh2(goc, out)
    hinh3(goc, out)
    hinh5(goc, out)
    hinh4(goc, out)
    print("Xong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
