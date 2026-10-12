"""Mô hình chuỗi: DLinear và GRU, dùng chung cho máy tại chỗ và cho Colab.

    # tại chỗ, môi trường ảo riêng có torch
    python scripts/dl.py --env E2 --horizons 1 --models dlinear --seeds 42
    # chuyển giao
    ... scripts/dl.py --cap E1:E2 --horizons 1 --models dlinear --seeds 42
    # trên Colab
    !python dl.py --env E1,E2 --horizons 1,12 --models gru --seeds 42 --goi goi_colab

Chuẩn hoá đã chốt
-----------------

Hai mô hình dùng **đúng một cách chuẩn hoá: N1, tức thống kê phần train của từng
máy**, giống hệt các mô hình bảng. Đầu vào là cửa sổ `L` điểm đã z-score, mục tiêu là
`y(t+h)` đã z-score bằng cùng `mu` và `sd`, và dự đoán đưa về CPU% bằng
`ŷ = ẑ·sd + mu`. **Không** dùng chuẩn hoá theo từng cửa sổ: hai cách cho hai nghĩa
khác nhau và không được mang chung một nhãn. Bản chạy thử ngày 20/09 dùng chuẩn hoá
theo cửa sổ, nên số đo thời gian của nó dùng được còn số sai số thì không.

Tập dòng
--------

Dòng huấn luyện và dòng chấm lấy từ `data/eval_rows/{env}_h{h}.parquet`, cột `bang`
và `chuoi`: cùng luật dòng hợp lệ với mô hình bảng, cộng thêm điều kiện cửa sổ `L`
điểm không có `NaN`. Nhờ vậy mô hình chuỗi và mô hình bảng chấm trên cùng một tập
dòng, và số dòng bị loại vì cửa sổ dài được báo ở `data/eval_rows/tom_tat.csv`.

Dừng sớm
--------

Huấn luyện trên vùng train, theo dõi trên vùng validation, dừng khi không cải thiện
sau `--patience` epoch, rồi **khớp lại** bằng đúng số epoch tốt nhất trên train +
validation. Cách này giữ đúng quy ước của các mô hình bảng: chọn cấu hình trên
validation, mô hình cuối học trên train + validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Trên máy Windows đang dùng, torch phải được import TRƯỚC numpy và pandas, nếu không
# thì nạp c10.dll hỏng với WinError 1114.
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

L_MAC_DINH = 288
KHUNG = {"n_train": 1612, "n_val": 345, "window": 2304}
COT_DU_DOAN = ["nguon", "dich", "mode", "lich", "model", "loss", "h", "fold",
               "seed", "series_id", "bucket_phat", "bucket_muc_tieu", "y", "yhat"]


# --------------------------------------------------------------- mô hình

class DLinear(nn.Module):
    """Tách xu thế bằng trung bình trượt, hai lớp tuyến tính, cộng lại."""

    def __init__(self, dai: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.xu_the = nn.Linear(dai, 1)
        self.mua_vu = nn.Linear(dai, 1)

    def forward(self, x):
        pad = self.kernel // 2
        m = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate"),
            kernel_size=self.kernel, stride=1).squeeze(1)
        return (self.xu_the(m) + self.mua_vu(x - m)).squeeze(1)


class GRUNho(nn.Module):
    """Một lớp GRU rồi một lớp tuyến tính, lấy trạng thái ẩn cuối cùng."""

    def __init__(self, an: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=an, batch_first=True)
        self.ra = nn.Linear(an, 1)

    def forward(self, x):
        o, _ = self.gru(x.unsqueeze(-1))
        return self.ra(o[:, -1]).squeeze(1)


def dung_model(ten: str, L: int) -> nn.Module:
    if ten == "dlinear":
        return DLinear(L)
    if ten == "gru":
        return GRUNho()
    raise SystemExit(f"Mô hình không rõ: {ten!r}. Có: dlinear, gru.")


# ----------------------------------------------------------------- dữ liệu

class MoiTruong:
    """Ma trận `y` theo (chuỗi, bucket), thống kê N1 và danh sách dòng hợp lệ."""

    def __init__(self, env: str, h: int, goc: Path, L: int):
        proc = pd.read_parquet(goc / "processed" / f"{env}.parquet",
                               columns=["series_id", "bucket", "y"])
        proc = proc.sort_values(["series_id", "bucket"])
        self.env, self.h, self.L = env, h, L
        self.b0 = int(proc["bucket"].min())
        self.ids = proc["series_id"].drop_duplicates().to_numpy()
        n, m = len(self.ids), len(proc) // len(self.ids)
        if len(proc) != n * m:
            raise SystemExit(f"{env}: lưới không đều.")
        self.m = m
        self.Y = proc["y"].to_numpy("float32").reshape(n, m)
        self.vi_tri = {s: i for i, s in enumerate(self.ids)}

        tr = self.Y[:, :KHUNG["n_train"]]
        with np.errstate(invalid="ignore"):
            self.mu = np.nanmean(tr, axis=1)
            self.sd = np.nanstd(tr, axis=1, ddof=1)
        self.xau = ~np.isfinite(self.sd) | (self.sd <= 0)
        self.sd = np.where(self.xau, np.nan, self.sd)

        er = pd.read_parquet(goc / "eval_rows" / f"{env}_h{h}.parquet")
        er = er[er["bang"] & er["chuoi"]]
        hang = np.array([self.vi_tri[s] for s in er["series_id"].to_numpy()])
        cot = er["bucket"].to_numpy("int64") - self.b0
        giu = ~self.xau[hang]
        self.hang, self.cot = hang[giu], cot[giu]

    def loc(self, lo: int, hi: int) -> np.ndarray:
        """Chỉ số các dòng có `lo <= t` và `t + h < hi`."""
        return np.flatnonzero((self.cot >= lo) & (self.cot + self.h < hi))

    def batch(self, ix: np.ndarray):
        h_, c_ = self.hang[ix], self.cot[ix]
        cua = np.stack([self.Y[h_[k], c_[k] - self.L + 1: c_[k] + 1]
                        for k in range(len(h_))])
        dich = self.Y[h_, c_ + self.h]
        mu = self.mu[h_].astype("float32")
        sd = self.sd[h_].astype("float32")
        x = (cua - mu[:, None]) / sd[:, None]
        z = (dich - mu) / sd
        return x.astype("float32"), z.astype("float32"), mu, sd, dich


def tensor(x, thiet_bi):
    return torch.from_numpy(x).to(thiet_bi)


# -------------------------------------------------------------- huấn luyện

def mot_epoch(model, mt: MoiTruong, ix: np.ndarray, opt, mat, batch: int,
              rng, thiet_bi, gioi_han: int = 0) -> float:
    model.train()
    thu_tu = rng.permutation(len(ix))
    tong, dem = 0.0, 0
    for b in range(0, len(thu_tu), batch):
        lo = ix[thu_tu[b: b + batch]]
        x, z, _, _, _ = mt.batch(lo)
        opt.zero_grad()
        ra = model(tensor(x, thiet_bi))
        loss = mat(ra, tensor(z, thiet_bi))
        loss.backward()
        opt.step()
        tong += float(loss.detach().cpu()) * len(lo)
        dem += len(lo)
        if gioi_han and (b // batch) + 1 >= gioi_han:
            break
    return tong / max(dem, 1)


@torch.no_grad()
def du_doan(model, mt: MoiTruong, ix: np.ndarray, batch: int, thiet_bi):
    model.eval()
    ra, that, sid, bk = [], [], [], []
    for b in range(0, len(ix), batch):
        lo = ix[b: b + batch]
        x, _, mu, sd, dich = mt.batch(lo)
        z = model(tensor(x, thiet_bi)).cpu().numpy()
        ra.append(z * sd + mu)
        that.append(dich)
        sid.append(mt.ids[mt.hang[lo]])
        bk.append(mt.cot[lo] + mt.b0)
    return (np.concatenate(ra), np.concatenate(that),
            np.concatenate(sid), np.concatenate(bk))


def mae_p50(sid, y, yhat) -> float:
    d = pd.DataFrame({"s": sid, "e": np.abs(np.asarray(y) - np.asarray(yhat))})
    return float(d.groupby("s")["e"].mean().median())


def chay_mot_to_hop(ten: str, mt_nguon: MoiTruong, mt_dich: MoiTruong, seed: int,
                    a, thiet_bi) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    tr = mt_nguon.loc(0, KHUNG["n_train"])
    va = mt_nguon.loc(KHUNG["n_train"], KHUNG["n_train"] + KHUNG["n_val"])
    fit = mt_nguon.loc(0, KHUNG["n_train"] + KHUNG["n_val"])
    te = mt_dich.loc(KHUNG["n_train"] + KHUNG["n_val"], KHUNG["window"])

    model = dung_model(ten, mt_nguon.L).to(thiet_bi)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    mat = nn.L1Loss()

    t0 = time.time()
    tot_nhat, epoch_tot, kien_nhan = float("inf"), 0, 0
    lich_su = []
    for ep in range(a.epochs):
        mat_tb = mot_epoch(model, mt_nguon, tr, opt, mat, a.batch, rng, thiet_bi,
                           a.gioi_han_batch)
        yh, y, sid, _ = du_doan(model, mt_nguon, va, a.batch_cham, thiet_bi)
        diem = mae_p50(sid, y, yh)
        lich_su.append({"epoch": ep, "loss_train": round(mat_tb, 5),
                        "mae_p50_val": round(diem, 5),
                        "giay": round(time.time() - t0, 1)})
        print(f"    epoch {ep:>2} loss={mat_tb:.4f} MAE_val={diem:.4f} "
              f"({time.time() - t0:.0f}s)")
        if diem < tot_nhat - 1e-6:
            tot_nhat, epoch_tot, kien_nhan = diem, ep + 1, 0
        else:
            kien_nhan += 1
            if kien_nhan >= a.patience:
                break
    giay_chon = time.time() - t0

    # Khớp lại trên train + validation bằng đúng số epoch tốt nhất.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = dung_model(ten, mt_nguon.L).to(thiet_bi)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t1 = time.time()
    for ep in range(max(epoch_tot, 1)):
        mot_epoch(model, mt_nguon, fit, opt, mat, a.batch, rng, thiet_bi,
                  a.gioi_han_batch)
    yh, y, sid, bk = du_doan(model, mt_dich, te, a.batch_cham, thiet_bi)

    du = pd.DataFrame({
        "nguon": mt_nguon.env, "dich": mt_dich.env, "mode": "N1", "lich": "khong",
        "model": ten, "loss": "tuyet_doi", "h": mt_dich.h, "fold": "test",
        "seed": seed, "series_id": sid, "bucket_phat": bk,
        "bucket_muc_tieu": bk + mt_dich.h, "y": y, "yhat": yh,
    })[COT_DU_DOAN]
    tep = (f"{mt_nguon.env}-{mt_dich.env}_N1_khong_{ten}_tuyet_doi_"
           f"h{mt_dich.h}_test_seed{seed}.parquet")
    (Path(a.out) / "du_doan").mkdir(parents=True, exist_ok=True)
    du.to_parquet(Path(a.out) / "du_doan" / tep, index=False)

    return {
        "thi_nghiem": "tn_a" if mt_nguon.env == mt_dich.env else "tn_c",
        "nguon": mt_nguon.env, "dich": mt_dich.env, "mode": "N1",
        "h": mt_dich.h, "model": ten, "loss": "tuyet_doi", "seed": seed,
        "fold": "test", "n_train": int(len(fit)), "n_cham": int(len(te)),
        "epoch_tot": epoch_tot, "mae_p50_val": round(tot_nhat, 5),
        "giay_chon": round(giay_chon, 1),
        "giay_khop_lai": round(time.time() - t1, 1),
        "tep_du_doan": tep, "thiet_bi": str(thiet_bi),
        "lich_su": json.dumps(lich_su, ensure_ascii=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default=None, help="danh sách môi trường tại chỗ")
    ap.add_argument("--cap", default=None, help="chuyển giao, dạng E1:E2,E2:E1")
    ap.add_argument("--horizons", default="1,12")
    ap.add_argument("--models", default="dlinear")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--goi", default="data", help="thư mục chứa processed/ và eval_rows/")
    ap.add_argument("--out", default="results/dl")
    ap.add_argument("--L", type=int, default=L_MAC_DINH)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--batch-cham", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gioi-han-batch", type=int, default=0,
                    help="chỉ chạy bấy nhiêu batch mỗi epoch, dùng khi đo chi phí")
    ap.add_argument("--thiet-bi", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--bat-buoc-gpu", action="store_true",
                    help="dừng nếu không có GPU, dùng trên Colab")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if a.thiet_bi == "auto":
        thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        thiet_bi = torch.device(a.thiet_bi)
    if a.bat_buoc_gpu and thiet_bi.type != "cuda":
        raise SystemExit(
            "Không thấy GPU. Trên Colab chọn Runtime > Change runtime type > GPU. "
            "Chạy GRU bằng CPU mất hàng chục giờ nên script dừng ở đây."
        )

    hs = [int(x) for x in a.horizons.split(",")]
    models = [x.strip() for x in a.models.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    cap = ([tuple(x.split(":")) for x in a.cap.split(",")] if a.cap
           else [(e.strip(), e.strip()) for e in (a.env or "").split(",") if e.strip()])
    if not cap:
        raise SystemExit("Cần --env hoặc --cap.")

    goc = Path(a.goi)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tom_tat = out / "tom_tat.csv"
    da_co = set()
    if a.resume and tom_tat.exists():
        d = pd.read_csv(tom_tat)
        da_co = {tuple(map(str, r)) for r in
                 d[["nguon", "dich", "h", "model", "seed"]].to_numpy()}

    print()
    print("MÔ HÌNH CHUỖI — DLinear và GRU")
    print(f"python  : {sys.executable}")
    print(f"torch   : {torch.__version__} · thiết bị {thiet_bi}")
    if thiet_bi.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"cặp {cap} · tầm {hs} · mô hình {models} · seed {seeds}")
    print(f"cửa sổ đầu vào L = {a.L} · epoch tối đa {a.epochs} · patience {a.patience}")
    print()

    dong = []
    for nguon, dich in cap:
        for h in hs:
            mt_n = MoiTruong(nguon, h, goc, a.L)
            mt_d = mt_n if dich == nguon else MoiTruong(dich, h, goc, a.L)
            print(f"{nguon}→{dich} h={h}: {len(mt_n.loc(0, KHUNG['n_train'])):,} "
                  f"dòng train · {len(mt_d.loc(KHUNG['n_train'] + KHUNG['n_val'], KHUNG['window'])):,} dòng test")
            for ten in models:
                for seed in seeds:
                    if (nguon, dich, str(h), ten, str(seed)) in da_co:
                        print(f"  {ten} seed {seed}: bỏ qua")
                        continue
                    print(f"  {ten} seed {seed}")
                    r = chay_mot_to_hop(ten, mt_n, mt_d, seed, a, thiet_bi)
                    dong.append(r)
                    df = pd.DataFrame(dong)
                    if tom_tat.exists():
                        df = pd.concat([pd.read_csv(tom_tat), df], ignore_index=True)
                        df = df.drop_duplicates(
                            subset=["nguon", "dich", "h", "model", "seed"],
                            keep="last")
                    df.to_csv(tom_tat, index=False)
                    print(f"    → epoch tốt {r['epoch_tot']} · "
                          f"MAE_val {r['mae_p50_val']} · "
                          f"{r['giay_chon'] + r['giay_khop_lai']:.0f}s")

    (out / "meta.json").write_text(json.dumps({
        "ngay": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:], "python": sys.executable,
        "torch": torch.__version__, "thiet_bi": str(thiet_bi),
        "cap": [list(c) for c in cap], "horizons": hs, "models": models,
        "seeds": seeds, "L": a.L, "epochs": a.epochs, "patience": a.patience,
        "batch": a.batch, "lr": a.lr,
        "chuan_hoa": "N1: mu, sd của phần train từng máy",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nXong. Kết quả ở {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
