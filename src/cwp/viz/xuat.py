"""Xuất hình theo hai dạng, tách bạch mục đích.

- **Mỗi panel một tệp ảnh riêng** (`.png`). Đúng **một hình** trong một tệp, tiêu đề
  mô tả thuần, không chữ diễn giải. Đây là tệp để dán thẳng vào bài báo hoặc slide:
  chọn được từng hình một, không phải cắt từ ảnh ghép.
- **Một tệp `.pdf` gộp**: trang đầu là diễn giải (đọc hình thế nào, phương pháp,
  cảnh báo), rồi mỗi panel một trang. Đây là tệp để đọc và để duyệt.

Vì sao tách. Ảnh ghép 2×2 tiện xem nhưng dở khi dùng lại: muốn đưa một panel vào bài
thì phải cắt ảnh, mà cắt ảnh raster thì mất nét và không sửa được. Còn chữ diễn giải
in cứng vào ảnh thì không sửa được khi câu chữ của bài đổi, và người đọc bài báo
không cần biết mã quyết định nội bộ của dự án.

Dùng ở `scripts/fig_*.py`. Mỗi script khai báo danh sách panel rồi gọi `xuat_bo_hinh`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NEN = "#fcfcfb"
MUC = "#0b0b0b"
MUC_PHU = "#52514e"


class Panel(NamedTuple):
    """Một hình đơn.

    ten      : tên tệp, không đuôi. Đặt số thứ tự ở đầu để thư mục tự sắp đúng thứ tự.
    tieu_de  : tiêu đề **mô tả thuần**. Kết luận thuộc về caption, không thuộc về hình.
    figsize  : cỡ khi đứng một mình, tính bằng inch.
    ve       : hàm nhận một `Axes` và vẽ vào đó. Không được tự tạo figure.
    """

    ten: str
    tieu_de: str
    figsize: tuple[float, float]
    ve: Callable


def chu_giai_duoi(ax, *, ncol: int = 3, fontsize: float = 8.2, **kw):
    """Chú giải đặt DƯỚI trục, nằm ngang.

    Khi mỗi panel đứng một mình thì không còn chỗ trống nào chắc chắn bên trong
    trục: vị trí "góc trên phải" vốn trống trong ảnh ghép lại rơi trúng dữ liệu ở
    tỉ lệ khung mới. Canh tay từng hình thì sẽ lệch lại ngay khi dữ liệu đổi. Đặt
    bên dưới thì **không bao giờ che dữ liệu**, và đó cũng là chỗ quen thuộc của
    chú giải trong hình bài báo.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=ncol,
                     fontsize=fontsize, frameon=False, labelcolor=MUC_PHU, **kw)


def _khung_hinh(figsize, nen):
    fig, ax = plt.subplots(figsize=figsize, facecolor=nen)
    ax.set_facecolor(nen)
    return fig, ax


def luu_tung_panel(panels: Iterable[Panel], thu_muc: Path, *,
                   dpi: int = 200, nen: str = NEN, duoi: str = "png") -> list[Path]:
    """Mỗi panel một tệp ảnh, đúng một hình trong tệp."""
    thu_muc.mkdir(parents=True, exist_ok=True)
    ra = []
    for p in panels:
        fig, ax = _khung_hinh(p.figsize, nen)
        p.ve(ax)
        ax.set_title(p.tieu_de, fontsize=10.5, color=MUC, loc="left", pad=10)
        dich = thu_muc / f"{p.ten}.{duoi}"
        fig.savefig(dich, dpi=dpi, facecolor=nen, bbox_inches="tight")
        plt.close(fig)
        ra.append(dich)
    return ra


def _trang_chu(pdf: PdfPages, tieu_de: str, phu: str, doan: list[tuple[str, str]],
               *, figsize: tuple[float, float], nen: str) -> None:
    """Trang diễn giải: tiêu đề, một câu phụ, rồi các đoạn có tiêu đề nhỏ."""
    fig = plt.figure(figsize=figsize, facecolor=nen)
    fig.text(0.06, 0.945, tieu_de, fontsize=16, color=MUC, ha="left", va="top")
    y = 0.90
    if phu:
        chu = textwrap.fill(phu, width=96)
        fig.text(0.06, y, chu, fontsize=9.5, color=MUC_PHU, ha="left", va="top",
                 linespacing=1.5)
        y -= 0.030 * (chu.count("\n") + 1) + 0.028

    for dau, than in doan:
        if y < 0.08:      # hết trang thì sang trang mới, không đè chữ lên nhau
            pdf.savefig(fig, facecolor=nen)
            plt.close(fig)
            fig = plt.figure(figsize=figsize, facecolor=nen)
            y = 0.945
        if dau:
            fig.text(0.06, y, dau, fontsize=10.5, color=MUC, ha="left", va="top",
                     fontweight="bold")
            y -= 0.034
        chu = textwrap.fill(than, width=96)
        fig.text(0.06, y, chu, fontsize=9.5, color=MUC_PHU, ha="left", va="top",
                 linespacing=1.5)
        y -= 0.0285 * (chu.count("\n") + 1) + 0.026

    pdf.savefig(fig, facecolor=nen)
    plt.close(fig)


def gop_pdf(panels: Iterable[Panel], dich: Path, *, tieu_de: str, phu: str,
            dien_giai: list[tuple[str, str]], chu_thich: dict[str, str] | None = None,
            trang: tuple[float, float] = (11.0, 8.5), nen: str = NEN) -> Path:
    """PDF nhiều trang: trang diễn giải, rồi mỗi panel một trang kèm chú thích."""
    dich.parent.mkdir(parents=True, exist_ok=True)
    chu_thich = chu_thich or {}
    panels = list(panels)

    with PdfPages(dich) as pdf:
        _trang_chu(pdf, tieu_de, phu, dien_giai, figsize=trang, nen=nen)

        for i, p in enumerate(panels, start=1):
            fig = plt.figure(figsize=trang, facecolor=nen)
            fig.text(0.06, 0.955, f"Hình {i}. {p.tieu_de}", fontsize=13, color=MUC,
                     ha="left", va="top")
            cap = chu_thich.get(p.ten, "")
            cao = 0.74 if not cap else 0.62
            ax = fig.add_axes([0.09, 0.94 - cao, 0.86, cao], facecolor=nen)
            p.ve(ax)
            if cap:
                fig.text(0.06, 0.10, textwrap.fill(cap, width=104), fontsize=9.5,
                         color=MUC_PHU, ha="left", va="top", linespacing=1.5)
            pdf.savefig(fig, facecolor=nen)
            plt.close(fig)

    return dich


def xuat_bo_hinh(panels: Iterable[Panel], thu_muc: Path, ten_pdf: str, *,
                 tieu_de: str, phu: str, dien_giai: list[tuple[str, str]],
                 chu_thich: dict[str, str] | None = None,
                 dpi: int = 200, nen: str = NEN) -> tuple[list[Path], Path]:
    """Xuất cả hai dạng. Trả về (danh sách tệp ảnh, đường dẫn PDF)."""
    panels = list(panels)
    anh = luu_tung_panel(panels, thu_muc, dpi=dpi, nen=nen)
    pdf = gop_pdf(panels, thu_muc / ten_pdf, tieu_de=tieu_de, phu=phu,
                  dien_giai=dien_giai, chu_thich=chu_thich, nen=nen)
    return anh, pdf


__all__ = ["Panel", "luu_tung_panel", "gop_pdf", "xuat_bo_hinh",
           "chu_giai_duoi", "NEN", "MUC", "MUC_PHU"]
