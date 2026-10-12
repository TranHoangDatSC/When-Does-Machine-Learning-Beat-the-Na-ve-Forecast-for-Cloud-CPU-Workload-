"""Độ giật (burstiness): hệ số biến thiên CV của từng chuỗi, ba môi trường.

    python scripts/fig_burstiness.py --env all

Sinh vào `results/figures/kham_pha/`: bốn tệp `.png` **mỗi tệp đúng một hình**
(`08_cv-ecdf`, `09_cv-so-voi-muc-tai`, `10_cv-trung-vi-iqr`, `11_cv-bang-so`) và
`kham-pha_burstiness.pdf` gộp cả bốn kèm trang diễn giải. Bảng số ở
`results/tables/cv.csv` (một dòng mỗi chuỗi, dùng lại để phân tầng khi báo cáo
Thí nghiệm 1) và `cv_summary.csv`.

## Định nghĩa

`CV = std / mean` của từng chuỗi, tính trên **mọi điểm không NaN** trong cửa sổ 8
ngày, `ddof = 1`.

Lưu ý một chỗ không đồng nhất: `catalog.mean` và `catalog.std` tính **chỉ trên điểm
quan sát thật**, còn CV ở đây tính trên **cả điểm nội suy**. Chênh lệch nằm ở chữ số
thập phân thứ tư vì tỉ lệ nội suy dưới 0,17%. Ghi ra để khỏi tưởng là lỗi.

## Vì sao trung vị và IQR, không phải trung bình

Bộ lọc `gan_chet` đã bỏ mọi chuỗi có mean dưới 1,0 nên CV không nổ. Nhưng một chuỗi
mean 1,01 vẫn kéo trung bình đi rất xa: đo được **trung bình CV cao hơn trung vị
+46% (E1), +76% (E2), +17% (E3)**. Báo cáo trung bình sẽ nói sai về môi trường nào
bursty hơn.

## Bốn panel

- (a) ECDF của CV, trục hoành log: phân phối đầy đủ, thấy được cả chồng lấn.
- (b) **CV so với mức tải trung bình của chuỗi.** Nếu CV chỉ là mức tải đổi tên thì
  phân tầng theo CV chẳng khác gì phân tầng theo mức tải, và không tách được ảnh hưởng
  của độ giật khỏi ảnh hưởng của mức tải.
- (c) Trung vị và IQR.
- (d) Bảng số.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # ưu tiên src của chính repo này

from cwp.evaluation.splits import doc_b0
from cwp.evaluation.tang import bang_cv
from cwp.viz import xuat

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]

NHAN = {
    "E1": "E1 — Bitbrains fastStorage (VM)",
    "E2": "E2 — Bitbrains Rnd 2013-8 (VM)",
    "E3": "E3 — Alibaba v2018 (máy vật lý)",
}
MAU = {"E1": "#2a78d6", "E2": "#eb6834", "E3": "#1baf7a"}
NET = {"E1": "-", "E2": "--", "E3": "-."}

MUC = "#0b0b0b"
MUC_PHU = "#52514e"
MUC_MO = "#8a8a85"
NEN = "#fcfcfb"


def do_cv(env: str, proc_dir: Path, cat: pd.DataFrame) -> pd.DataFrame:
    """Một dòng mỗi chuỗi: CV trên cả cửa sổ và CV trên phần train.

    Công thức nằm ở `cwp.evaluation.tang.bang_cv`, dùng chung với
    `scripts/build_may_gia.py`.
    """
    f = proc_dir / f"{env}.parquet"
    if not f.exists():
        raise SystemExit(f"Thiếu {f}. Chạy: python -m cwp.preprocess.build --env all")

    d = pd.read_parquet(f, columns=["series_id", "bucket", "y"])
    giu = set(cat[(cat["env"] == env) & cat["kept"]]["series_id"])
    if set(d["series_id"].unique()) != giu:
        raise SystemExit(
            f"{env}: data/processed/ không khớp catalog.kept. Sinh lại cả hai bằng "
            "một lệnh: python -m cwp.preprocess.build --env all"
        )

    return bang_cv(d, env, doc_b0(env, proc_dir))


def tom_tat(cv: pd.DataFrame) -> dict:
    v = cv["cv"].dropna().to_numpy()
    q = np.percentile(v, [5, 25, 50, 75, 95])
    rho, p = spearmanr(cv["cv"], cv["mean"])
    return {
        "env": cv["env"].iloc[0],
        "n_chuoi": len(v),
        "cv_p5": q[0], "cv_p25": q[1], "cv_p50": q[2], "cv_p75": q[3], "cv_p95": q[4],
        "cv_iqr": q[3] - q[1],
        "cv_iqr_tren_p50": (q[3] - q[1]) / q[2],
        "cv_min": v.min(), "cv_max": v.max(),
        "cv_trung_binh": v.mean(),
        "tb_lech_p50_pct": 100.0 * (v.mean() / q[2] - 1.0),
        "spearman_cv_mean": rho,
        "spearman_p": p,
        "cham_chan_p50": float(cv["ti_le_cham_chan"].median()),
        "sat_chan_pct": 100.0 * float((cv["ti_le_cham_chan"] >= 0.8).mean()),
        "vi_pham_chan": int((cv["ti_le_cham_chan"] > 1.0001).sum()),
    }


# ------------------------------------------------------------------- vẽ

def khung(ax, *, xlabel, ylabel):
    ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    for c in ("left", "bottom"):
        ax.spines[c].set_color("#d4d3ce")
    ax.tick_params(colors=MUC_PHU, labelsize=8, length=3)
    ax.set_xlabel(xlabel, fontsize=9, color=MUC_PHU)
    ax.set_ylabel(ylabel, fontsize=9, color=MUC_PHU)


def panel_ecdf(ax, cv):
    """ECDF của CV, trục hoành log."""
    for env in ENVS:
        v = np.sort(cv[env]["cv"].dropna().to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=MAU[env],
                linestyle=NET[env], linewidth=1.8, label=NHAN[env], zorder=3)
    ax.axhline(0.5, color=MUC_MO, linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 10)
    ax.set_ylim(-0.02, 1.02)
    _so_gon_truc(ax, "x")
    khung(ax, xlabel="CV = std / mean của chuỗi", ylabel="Tỉ lệ chuỗi tích luỹ")
    ax.text(0.034, 0.52, "trung vị", fontsize=7.5, color=MUC_MO, va="bottom")
    xuat.chu_giai_duoi(ax)


def panel_cv_vs_tai(ax, cv, s):
    """CV so với mức tải, kèm chặn của thang đo."""
    for env in ENVS:
        ax.scatter(cv[env]["mean"], cv[env]["cv"], s=7, color=MAU[env], alpha=0.42,
                   linewidths=0, zorder=3, label=NHAN[env])
    m = np.geomspace(1.0, 99.0, 400)
    ax.plot(m, np.sqrt((100.0 - m) / m), color=MUC, linewidth=1.1,
            linestyle=(0, (5, 2)), zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    khung(ax, xlabel="CPU% trung bình của chuỗi", ylabel="CV")
    for f in ("x", "y"):
        _so_gon_truc(ax, f)
    # Vùng phía TRÊN đường chặn rỗng theo định nghĩa, nên chú thích ở đó không đè
    # lên điểm nào.
    ax.text(26, 4.2, "chặn CV = √((100−m)/m)\ndo thang đo 0–100,\n"
            "không phải do dữ liệu",
            fontsize=7.8, color=MUC, ha="left", va="center", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=NEN, edgecolor="#d4d3ce",
                      linewidth=0.7), zorder=5)
    ax.text(0.02, 0.03,
            "\n".join(f"{e}: Spearman ρ = {s.loc[e, 'spearman_cv_mean']:+.3f}"
                      for e in ENVS),
            transform=ax.transAxes, fontsize=8.2, color=MUC_PHU, va="bottom",
            ha="left", bbox=dict(boxstyle="round,pad=0.45", facecolor=NEN,
                                 edgecolor="#d4d3ce", linewidth=0.7))
    # Khi panel đứng một mình thì nó phải tự mang chú giải; trong bản ghép trước đây
    # nó mượn chú giải của panel kề bên. markerscale phóng to chấm vì chấm dữ liệu
    # cỡ 7pt quá nhỏ để đọc trong ô chú giải.
    xuat.chu_giai_duoi(ax, markerscale=3.0, scatterpoints=1)


def panel_trung_vi_iqr(ax, s):
    """Trung vị và IQR — thanh đậm p25–p75, vạch mảnh p5–p95."""
    for i, env in enumerate(ENVS):
        y = len(ENVS) - 1 - i
        r = s.loc[env]
        ax.plot([r["cv_p5"], r["cv_p95"]], [y, y], color=MAU[env], linewidth=1.2,
                alpha=0.55, zorder=2)
        ax.plot([r["cv_p25"], r["cv_p75"]], [y, y], color=MAU[env], linewidth=9,
                solid_capstyle="butt", alpha=0.85, zorder=3)
        ax.plot([r["cv_p50"]], [y], marker="|", markersize=16, markeredgewidth=2.4,
                color=NEN, zorder=4)
        ax.text(r["cv_p95"] * 1.12, y, f"  trung vị {r['cv_p50']:.3f}  ·  "
                f"IQR {r['cv_iqr']:.3f}", fontsize=8.2, color=MUC_PHU,
                va="center", ha="left")
    ax.set_yticks(range(len(ENVS)))
    ax.set_yticklabels(ENVS[::-1], fontsize=9)
    ax.set_ylim(-0.6, len(ENVS) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 22)
    _so_gon_truc(ax, "x")
    khung(ax, xlabel="CV (trục log)", ylabel="")


def panel_bang(ax, s):
    """Bảng số."""
    ax.axis("off")
    hang = [("n_chuoi", "Số chuỗi", "{:,.0f}"), ("cv_p25", "p25", "{:.4f}"),
            ("cv_p50", "Trung vị", "{:.4f}"), ("cv_p75", "p75", "{:.4f}"),
            ("cv_iqr", "IQR", "{:.4f}"),
            ("cv_iqr_tren_p50", "IQR / trung vị", "{:.2f}"),
            ("cv_trung_binh", "Trung bình", "{:.4f}"),
            ("tb_lech_p50_pct", "TB lệch trung vị %", "{:+.0f}"),
            ("cv_max", "Lớn nhất", "{:.2f}"),
            ("spearman_cv_mean", "ρ(CV, mức tải)", "{:+.3f}"),
            ("cham_chan_p50", "CV / chặn, trung vị", "{:.3f}"),
            ("sat_chan_pct", "% chuỗi sát chặn", "{:.1f}")]
    o = [[f.format(s.loc[e, k]) for e in ENVS] for k, _, f in hang]
    t = ax.table(cellText=o, rowLabels=[n for _, n, _ in hang], colLabels=ENVS,
                 cellLoc="right", rowLoc="right", loc="center", colWidths=[0.2] * 3)
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.36)
    manh = {"cv_p50", "cv_iqr", "spearman_cv_mean", "cham_chan_p50"}
    for (r, c), o_ in t.get_celld().items():
        o_.set_edgecolor("#e6e5e0")
        o_.set_linewidth(0.6)
        if r == 0:
            o_.set_facecolor("#f2f1ec")
            o_.get_text().set_color(MAU[ENVS[c]] if c >= 0 else MUC)
            o_.get_text().set_fontweight("bold")
        else:
            dam = hang[r - 1][0] in manh
            o_.get_text().set_color(MUC if dam else MUC_PHU)
            if dam:
                o_.get_text().set_fontweight("bold")


def _so_gon_truc(ax, truc):
    g = getattr(ax, f"{truc}axis")
    g.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    g.set_minor_formatter(matplotlib.ticker.NullFormatter())


def panels(cv, tt) -> list:
    s = tt.set_index("env")
    return [
        xuat.Panel("08_cv-ecdf", "ECDF của hệ số biến thiên CV (trục log)",
                   (7.0, 4.8), lambda ax: panel_ecdf(ax, cv)),
        xuat.Panel("09_cv-so-voi-muc-tai", "CV so với CPU% trung bình của chuỗi",
                   (7.0, 4.8), lambda ax: panel_cv_vs_tai(ax, cv, s)),
        xuat.Panel("10_cv-trung-vi-iqr",
                   "CV: trung vị và IQR — thanh đậm p25–p75, vạch mảnh p5–p95",
                   (7.4, 3.4), lambda ax: panel_trung_vi_iqr(ax, s)),
        xuat.Panel("11_cv-bang-so", "Bảng số — hệ số biến thiên CV",
                   (6.4, 4.8), lambda ax: panel_bang(ax, s)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="all", choices=ENVS + ["all"])
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--catalog", default="data/catalog.parquet")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures/kham_pha")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    envs = ENVS if a.env == "all" else [a.env]
    out_dir, tab_dir = ROOT / a.out, ROOT / a.tables
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cat = pd.read_parquet(ROOT / a.catalog)
    cv = {e: do_cv(e, ROOT / a.processed, cat) for e in envs}
    tt = pd.DataFrame([tom_tat(cv[e]) for e in envs])

    hau_to = "" if a.env == "all" else f"_{a.env}"
    p_chuoi = tab_dir / f"cv{hau_to}.csv"
    p_tom = tab_dir / f"cv_summary{hau_to}.csv"
    pd.concat(cv.values(), ignore_index=True).round(6).to_csv(
        p_chuoi, index=False, encoding="utf-8")
    tt.round(6).to_csv(p_tom, index=False, encoding="utf-8")

    print()
    print("ĐỘ GIẬT: CV = std/mean từng chuỗi (ddof = 1)")
    print()
    s = tt.set_index("env")
    print(f"{'':<22}" + "".join(f"{e:>13}" for e in envs))
    for k, n, f in (("n_chuoi", "Số chuỗi", "{:,.0f}"), ("cv_p25", "p25", "{:.4f}"),
                    ("cv_p50", "Trung vị", "{:.4f}"), ("cv_p75", "p75", "{:.4f}"),
                    ("cv_iqr", "IQR", "{:.4f}"),
                    ("cv_iqr_tren_p50", "IQR / trung vị", "{:.2f}"),
                    ("cv_trung_binh", "Trung bình", "{:.4f}"),
                    ("tb_lech_p50_pct", "TB lệch trung vị %", "{:+.0f}"),
                    ("spearman_cv_mean", "ρ(CV, mức tải)", "{:+.3f}"),
                    ("cham_chan_p50", "CV / chặn, trung vị", "{:.3f}"),
                    ("sat_chan_pct", "% chuỗi sát chặn", "{:.1f}"),
                    ("vi_pham_chan", "Chuỗi vượt chặn", "{:.0f}")):
        print(f"{n:<22}" + "".join(f"{f.format(s.loc[e, k]):>13}" for e in envs))

    if len(envs) == len(ENVS):
        lo, hi = s.loc["E3", "cv_p25"], s.loc["E3", "cv_p75"]
        print(f"\nChồng lấn với IQR của E3 [{lo:.4f}, {hi:.4f}]:")
        for e in ("E1", "E2"):
            v = cv[e]["cv"].to_numpy()
            print(f"  {e}: {((v >= lo) & (v <= hi)).mean():.1%} trong khoảng | "
                  f"{(v > hi).mean():.1%} trên | {(v < lo).mean():.1%} dưới")

    plt.rcParams["font.family"] = "DejaVu Sans"
    anh, pdf = xuat.xuat_bo_hinh(
        panels(cv, tt), out_dir, "kham-pha_burstiness.pdf",
        tieu_de="Độ giật: hệ số biến thiên CV của từng chuỗi",
        phu=("CV = std / mean của từng chuỗi (ddof = 1), tính trên mọi điểm quan sát "
             "trong cửa sổ 8 ngày, cho các chuỗi còn lại sau bộ lọc. Sinh bằng "
             "`python scripts/fig_burstiness.py --env all`."),
        dien_giai=dien_giai(tt.set_index("env")),
        chu_thich=chu_thich_panel(tt.set_index("env")), dpi=a.dpi)

    print()
    for q in list(anh) + [pdf, p_chuoi, p_tom]:
        print(f"Đã ghi: {q.relative_to(ROOT)}")
    print("\nMỗi .png chứa đúng MỘT hình. Tệp .pdf gộp bốn hình kèm phần diễn giải.")
    return 0



def dien_giai(s) -> list[tuple[str, str]]:
    """Trang diễn giải của tệp PDF: CV và cái chặn của thang đo đều dễ đọc nhầm."""
    return [
        ("CV là gì",
         "Hệ số biến thiên của một chuỗi là độ lệch chuẩn chia cho trung bình của "
         "chính chuỗi đó. Nó đo mức dao động TƯƠNG ĐỐI: một máy chạy quanh 40% mà "
         "dao động ±4 điểm có CV 0,1, còn một máy chạy quanh 2% mà dao động ±2 điểm "
         "có CV 1,0: máy thứ hai giật hơn nhiều dù biên độ tuyệt đối nhỏ hơn. "
         "Chia cho trung bình chính là để so được giữa các máy có mức tải khác nhau."),
        ("Vì sao báo trung vị và IQR chứ không phải trung bình",
         "Bộ lọc đã bỏ mọi chuỗi có trung bình dưới 1,0 nên CV không nổ vô hạn. "
         "Nhưng một chuỗi trung bình 1,01 vẫn kéo giá trị trung bình đi rất xa: đo "
         f"được trung bình CV cao hơn trung vị {s.loc['E1','tb_lech_p50_pct']:+.0f}% "
         f"ở E1 và {s.loc['E2','tb_lech_p50_pct']:+.0f}% ở E2. Báo trung bình sẽ nói "
         "sai về môi trường nào giật hơn. IQR là khoảng giữa p25 và p75, đo độ "
         "TẢN của các chuỗi trong cùng một môi trường."),
        ("Cái chặn ở Hình 2 là toán, không phải dữ liệu",
         "CPU% nằm trong [0, 100]. Một biến bị chặn như vậy với trung bình m không "
         "thể có phương sai vượt m(100 − m), nên CV của nó không thể vượt "
         "căn bậc hai của (100 − m)/m. Đó là đường đứt nét ở Hình 2. Chặn siết rất "
         "chặt ở vùng tải cao: máy chạy quanh 40% thì CV không thể vượt 1,22 dù có "
         "biến động thế nào. Đo trên dữ liệu: không chuỗi nào trong 1.535 chuỗi vượt "
         "chặn, và rìa chéo của đám điểm trùng khít đường chặn."),
        ("Vì sao cái chặn đó đảo ngược cách đọc",
         "Nhìn CV thô thì E3 ít giật nhất (trung vị 0,286 so với 0,497 và 0,537). "
         "Nhưng E3 chạy ở mức tải quanh 40%, nơi chặn chỉ là 1,22; còn E1 và E2 chạy "
         "quanh 2,7%, nơi chặn tới 6,0. So với mức mà thang đo cho phép ở mức tải "
         "của chính nó, E3 dùng tới 0,233 lần chặn còn E1 chỉ 0,094, tức E3 dùng "
         "gấp hơn hai lần. CV thô của E3 thấp một phần chỉ vì nó chạy nóng."),
        ("Có dùng CV để phân tầng máy được không",
         "Trong cùng một môi trường thì được: E1 và E2 có độ tản rộng, thừa chỗ chia "
         "ba tầng. Xuyên môi trường bằng một ngưỡng chung thì không, vì hai lý do đo "
         f"được. Một, tương quan hạng giữa CV và mức tải đổi dấu: "
         f"{s.loc['E1','spearman_cv_mean']:+.3f} ở E1 và "
         f"{s.loc['E3','spearman_cv_mean']:+.3f} ở E3: ở Bitbrains máy tải cao thì "
         "giật hơn, ở Alibaba thì ngược lại, nên một ngưỡng chung sẽ chọn ra hai "
         "nhóm máy khác loại. Hai, tam phân vị của E3 chỉ rộng 0,06 và nằm gọn trong "
         "tầng thấp nhất của E1."),
    ]


def chu_thich_panel(s) -> dict[str, str]:
    return {
        "08_cv-ecdf":
            "Trục hoành log vì CV trải từ 0,05 tới hơn 7. Đường của E3 dựng đứng: "
            "498 máy hành xử gần như nhau; E1 và E2 thoải: mỗi VM một kiểu. Khoảng "
            "một phần ba chuỗi E1/E2 nằm DƯỚI khoảng IQR của E3, nên không thể nói "
            "gọn là Bitbrains giật hơn.",
        "09_cv-so-voi-muc-tai":
            "Mỗi chấm là một chuỗi. Đường đứt nét là chặn của thang đo, không phải "
            "xu hướng của dữ liệu. Hai đám điểm tách hẳn nhau theo trục hoành chính "
            "là chênh lệch mức tải giữa Bitbrains và Alibaba.",
        "10_cv-trung-vi-iqr":
            "Thanh đậm là khoảng p25–p75 (IQR), vạch mảnh là p5–p95, vạch trắng ở "
            "giữa là trung vị. Trục hoành log. Chiều dài thanh đậm là thứ đáng nhìn: "
            "E3 ngắn hơn E1 gần chín lần.",
        "11_cv-bang-so":
            "Dòng \"IQR / trung vị\" chuẩn hoá độ tản theo mức của chính môi trường. "
            "Dòng \"CV / chặn\" và \"% chuỗi sát chặn\" cho biết một CV thấp là do bản "
            "chất hay do đã chạm trần thang đo.",
    }

if __name__ == "__main__":
    sys.exit(main())
