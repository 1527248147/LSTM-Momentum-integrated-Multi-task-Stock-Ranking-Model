# -*- coding: utf-8 -*-
"""
给 alpha158+fund 的全年面板数据打标签（MiM-StocR 风格）：
1) label_ret_1d：次交易日收益率
2) label_mom_cls：动量线 5 分类（Bounce/Positive/Volatile/Negative/Sink）

关键要求（用户指定）：
- 任何 label 缺失都不能删行（LSTM 需要保持面板/序列完整）
- 若 t+1 停牌（volume==0），则 t 的 label_ret_1d 设为无效：NaN + mask=0
- 动量标签：
    - 若窗口内出现“缺行”（OHLCV 完全没有记录） -> 动量标签无效(-1)
    - 若窗口内出现“停牌”（volume==0，但有记录） -> 跳过该日向前找交易日补足
- 输出后自动检查：
    - 每年 ret/mom/both 的缺失率
    - 每年 ret=0 的比例（仅统计 valid）
    - 抽样复算动量分类一致性（mismatch rate + mismatch 列表）

"""

import os
import re
import json
import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# 配置区：按你本机路径修改
# ============================================================
# feature：alpha158 + 财务因子的 yearly parquet（你现在是 year=2005.parquet 这种）
FEATURE_DIR = r"C:\AI_STOCK\dataset\dl_alpha158_plus_fund_yearly_parquet\dl_alpha158_plus_fund_yearly_parquet"

# OHLCV：RiceQuant 拉取的 yearly parquet（你现在是 ohlcv_2005.parquet 这种）
OHLCV_DIR = r"C:\AI_STOCK\dataset\ohlcv_ricequant_2004-2025_parquet_suspension_false"

# 交易日历
CAL_PATH = r"C:\AI_STOCK\project_alpha158+ricequant_fin+lgbm\dataset\trading_calendar_from_merged.csv"

# 输出目录（默认在脚本目录下）
OUT_DIR = os.path.join(os.path.dirname(__file__), "alpha158_plus_fund_yearly_parquet_labeled_mimstocr")
os.makedirs(OUT_DIR, exist_ok=True)

# 动量参数（与你当前脚本一致）
GAP_L = 4        # 5日动量差（交易日意义上）
LINE_LEN = 7     # 动量线长度（论文默认 s=6 => 7点）

# 为了处理停牌跳过/跨年 next-day，给每年额外往前/往后取一点交易日
CAL_BUFFER = 80

# ret=0 判断阈值（float32 下很少需要，但安全）
ZERO_EPS = 1e-12

# 动量审计抽样量（每年抽多少条 valid 样本复算）
AUDIT_MOM_SAMPLE_PER_YEAR = 200
AUDIT_SEED = 42


# ============================================================
# 工具函数：列名选择、日期与股票代码统一
# ============================================================
def pick_col(cols, candidates, required=True, name=""):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    if required:
        raise ValueError(f"找不到列 {name}。尝试：{candidates}。现有列：{list(cols)[:60]} ...")
    return None


def normalize_date_series(x: pd.Series) -> pd.Series:
    """把各种 date 格式统一到 pandas.Timestamp(date-level)"""
    if np.issubdtype(x.dtype, np.datetime64):
        return pd.to_datetime(x, errors="coerce").dt.normalize()
    if np.issubdtype(x.dtype, np.integer):
        return pd.to_datetime(x.astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
    return pd.to_datetime(x, errors="coerce").dt.normalize()


def normalize_instrument_series(s: pd.Series) -> pd.Series:
    """
    统一股票代码到：000001.SZ / 600000.SH / 430047.BJ
    兼容输入：
      - 000001.XSHE / 600000.XSHG
      - SZ000001 / SH600000
      - 000001.SZ / 600000.SH
      - 000001（按首位推断交易所：6/9->SH, 0/3->SZ, 4/8->BJ）
    """
    x = s.astype(str).str.upper().str.strip()
    x = x.str.replace(r"\s+", "", regex=True)

    # Qlib/RQ 后缀
    x = x.str.replace(".XSHG", ".SH", regex=False)
    x = x.str.replace(".XSHE", ".SZ", regex=False)

    # 前缀形式：SH600000 / SZ000001 / BJ430047
    ex_code = x.str.extract(r"^(SH|SZ|BJ)(\d{6})$")
    m = ex_code[0].notna()
    x.loc[m] = ex_code.loc[m, 1] + "." + ex_code.loc[m, 0]

    # 无点后缀：600000SH / 000001SZ
    ex_code2 = x.str.extract(r"^(\d{6})(SH|SZ|BJ)$")
    m2 = ex_code2[0].notna()
    x.loc[m2] = ex_code2.loc[m2, 0] + "." + ex_code2.loc[m2, 1]

    # 纯 6 位：推断交易所
    pure = x.str.fullmatch(r"\d{6}", na=False)
    if pure.any():
        code = x.loc[pure]
        first = code.str[0]
        exch = np.where(first.isin(["6", "9"]), "SH",
               np.where(first.isin(["0", "3"]), "SZ",
               np.where(first.isin(["4", "8"]), "BJ", "")))
        exch_s = pd.Series(exch, index=code.index)
        x.loc[pure] = code + "." + exch_s
        x.loc[pure & (exch_s == "")] = np.nan

    ok = x.str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)
    return x.where(ok, np.nan)


def _extract_year_from_name(name: str) -> int:
    m = re.search(r"(\d{4})", name)
    if not m:
        raise ValueError(f"无法从文件名提取年份：{name}")
    return int(m.group(1))


def find_year_parquet_files(root_dir: str, prefer_prefix: str = None):
    """
    递归扫描 root_dir 下所有包含 4 位年份的 parquet 文件。
    prefer_prefix：用于排序优先级（例如 feature 更偏好 year=，ohlcv 更偏好 ohlcv_）
    """
    files = []
    for root, _, fns in os.walk(root_dir):
        for fn in fns:
            if not fn.lower().endswith(".parquet"):
                continue
            if not re.search(r"\d{4}", fn):
                continue
            files.append(os.path.join(root, fn))

    if not files:
        return []

    def score(path):
        fn = os.path.basename(path).lower()
        y = _extract_year_from_name(fn)
        pref = 0
        if prefer_prefix:
            pref = 0 if fn.startswith(prefer_prefix.lower()) else 1
        return (y, pref, fn)

    files.sort(key=score)
    return files


def pick_best_year_file(files, year, allow_patterns=None):
    """从 files 中挑一个最像该 year 的文件（并按 allow_patterns 优先）"""
    cands = [p for p in files if _extract_year_from_name(os.path.basename(p)) == year]
    if not cands:
        return None

    if allow_patterns:
        for pat in allow_patterns:
            r = re.compile(pat, re.IGNORECASE)
            pri = [p for p in cands if r.search(os.path.basename(p))]
            if pri:
                pri.sort(key=lambda p: len(os.path.basename(p)))
                return pri[0]

    cands.sort(key=lambda p: len(os.path.basename(p)))
    return cands[0]


# ============================================================
# 动量分类逻辑（5 类）
# ============================================================
def classify_momentum_line(m_line: np.ndarray, eps: float = 1e-12) -> int:
    """
    5 类编码：
      4 = Bounce   (负 -> 正，单次穿越)
      3 = Positive (全为正)
      2 = Volatile (其它情况：多次穿越/混合/全零等)
      1 = Negative (全为负)
      0 = Sink     (正 -> 负，单次穿越)
    """
    s = np.sign(m_line)
    s[np.abs(m_line) <= eps] = 0

    if np.all(s > 0):
        return 3
    if np.all(s < 0):
        return 1

    nz = s.copy()
    # 前向填 0
    for i in range(1, len(nz)):
        if nz[i] == 0:
            nz[i] = nz[i - 1]
    # 后向填开头的 0
    for i in range(len(nz) - 2, -1, -1):
        if nz[i] == 0:
            nz[i] = nz[i + 1]

    if np.all(nz == 0):
        return 2

    changes = np.sum((nz[1:] * nz[:-1]) < 0)
    if changes == 1:
        if nz[0] < 0 and nz[-1] > 0:
            return 4
        if nz[0] > 0 and nz[-1] < 0:
            return 0
    return 2


def load_trading_calendar(cal_path: str) -> pd.Index:
    cal = pd.read_csv(cal_path)
    dcol = pick_col(cal.columns, ["date", "trade_date", "datetime"], name="交易日历日期")
    dates = normalize_date_series(cal[dcol]).dropna().sort_values().unique()
    return pd.Index(dates)


def build_date_window(cal_index: pd.Index, year: int, buffer_days: int):
    """给 year 构造一个带 buffer 的 date_list（跨年前后），用于 pivot 和 t+1"""
    mask = (cal_index.year == year)
    if not mask.any():
        raise ValueError(f"交易日历中找不到年份 {year}")
    year_dates = cal_index[mask]
    y_start = year_dates.min()
    y_end = year_dates.max()

    start_pos = cal_index.get_loc(y_start)
    end_pos = cal_index.get_loc(y_end)

    win_start = max(0, start_pos - buffer_days)
    win_end = min(len(cal_index) - 1, end_pos + 2)  # +2 保证 next-day

    date_list = cal_index[win_start:win_end + 1]
    return year_dates, date_list


def _arrow_filter_for_date(dataset, date_col: str, start: pd.Timestamp, end: pd.Timestamp):
    """给 pyarrow.dataset 生成日期范围 filter（兼容 int YYYYMMDD）"""
    field = ds.field(date_col)
    try:
        t = dataset.schema.field(date_col).type
    except Exception:
        t = None

    if t is not None and pa.types.is_integer(t):
        start_i = int(start.strftime("%Y%m%d"))
        end_i = int(end.strftime("%Y%m%d"))
        return (field >= start_i) & (field <= end_i)

    try:
        return (field >= pa.scalar(start)) & (field <= pa.scalar(end))
    except Exception:
        return None


def load_ohlcv_window(ohlcv_files: list, year: int, date_list: pd.Index) -> pd.DataFrame:
    """
    加载 date_list 覆盖到的所有年份 OHLCV，并只保留 date/instrument/close/volume
    """
    start = pd.Timestamp(date_list.min())
    end = pd.Timestamp(date_list.max())
    years_needed = sorted(set(date_list.year.tolist()))

    frames = []
    for y in years_needed:
        path = pick_best_year_file(
            ohlcv_files, y,
            allow_patterns=[r"^ohlcv_", r"^year=", r"^\d{4}\.parquet$"]
        )
        if path is None or (not os.path.exists(path)):
            continue

        dataset = ds.dataset(path, format="parquet")
        cols = dataset.schema.names

        dcol = pick_col(cols, ["date", "trade_date", "datetime"], name="OHLCV 日期")
        icol = pick_col(cols, ["instrument", "order_book_id", "code", "symbol", "ts_code"], name="OHLCV 股票列")
        ccol = pick_col(cols, ["close", "Close", "adj_close", "adjclose"], name="OHLCV close")
        vcol = pick_col(cols, ["volume", "total_volume", "vol", "成交量"], name="OHLCV volume")

        flt = _arrow_filter_for_date(dataset, dcol, start, end)
        try:
            table = dataset.to_table(columns=[dcol, icol, ccol, vcol], filter=flt) if flt is not None \
                    else dataset.to_table(columns=[dcol, icol, ccol, vcol])
            df = table.to_pandas()
        except Exception:
            table = dataset.to_table(columns=[dcol, icol, ccol, vcol])
            df = table.to_pandas()

        df.rename(columns={dcol: "date", icol: "instrument", ccol: "close", vcol: "volume"}, inplace=True)
        df["date"] = normalize_date_series(df["date"])
        df["instrument_norm"] = normalize_instrument_series(df["instrument"])

        # 仅清理 OHLCV 自身的脏行，不影响 feature 行
        df = df.dropna(subset=["date", "instrument_norm"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

        frames.append(df[["date", "instrument_norm", "close", "volume"]])

    if not frames:
        raise ValueError(
            f"Year {year}：未加载到任何 OHLCV 数据。"
            f"需要年份 {years_needed}，日期范围 [{start.date()}..{end.date()}]。请检查 OHLCV_DIR。"
        )

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv.sort_values(["instrument_norm", "date"], inplace=True)
    ohlcv.drop_duplicates(["instrument_norm", "date"], keep="last", inplace=True)
    return ohlcv


def compute_momentum_matrix(date_list: pd.Index, close_mat: np.ndarray, vol_mat: np.ndarray,
                            gap_l: int, line_len: int) -> np.ndarray:
    """
    计算整张动量分类矩阵 mom_cls，shape=(D,N)，值域 [-1,0..4]
    -1 表示无效（缺行/历史不足/无法构造）
    注意：
      - 缺行：close 是 NaN（pivot 没有记录），一旦窗口跨过缺行 => 无效
      - 停牌：volume<=0，但如果该日有记录（close 非 NaN），允许跳过向前找交易日补足
    """
    D, N = close_mat.shape
    mom = np.full((D, N), -1, dtype=np.int8)

    for j in tqdm(range(N), desc="计算动量分类(按股票)", ncols=95):
        close = close_mat[:, j]
        vol = vol_mat[:, j]

        present = ~np.isnan(close)  # 有记录（不管是否停牌）
        vol0 = np.where(np.isnan(vol), 0.0, vol)
        is_trade = present & (vol0 > 0)  # 可交易日（volume>0）

        # 缺行位置（OHLCV 没有这一天记录）
        missing = ~present
        # last_missing[p] = <=p 的最后一个缺行位置
        last_missing = np.maximum.accumulate(np.where(missing, np.arange(D), -1))

        trade_pos = np.flatnonzero(is_trade)
        if trade_pos.size == 0:
            continue

        trade_count = np.cumsum(is_trade)
        trade_rank = np.maximum.accumulate(np.where(is_trade, trade_count - 1, -1))

        for p in range(D):
            if not present[p]:
                continue

            r = trade_rank[p]
            if r < 0:
                continue

            # 历史是否足够
            if r - (line_len - 1) < 0:
                continue
            if (r - (line_len - 1) - gap_l) < 0:
                continue

            ranks = np.arange(r - (line_len - 1), r + 1, dtype=np.int32)
            t_list = trade_pos[ranks]

            ok = True
            m_line = np.empty(line_len, dtype=np.float32)

            for k, rr in enumerate(ranks):
                base_rank = rr - gap_l
                if base_rank < 0:
                    ok = False
                    break

                t0 = trade_pos[base_rank]

                # 若区间 (t0, t_list[k]] 出现缺行，则无效
                if last_missing[t_list[k]] > t0:
                    ok = False
                    break

                m_line[k] = close[t_list[k]] - close[t0]

            if not ok:
                continue

            mom[p, j] = classify_momentum_line(m_line)

    return mom


# ============================================================
# 核心：写入每年 labeled parquet（不删 feature 行）
# ============================================================
def write_labeled_year(feature_path: str, out_path: str,
                       cal_index: pd.Index,
                       ohlcv_files: list,
                       year: int):
    """
    给某一年的 feature parquet 打标签并写出 out_path
    重要：绝不因 label 缺失而删行
    """
    year_dates, date_list = build_date_window(cal_index, year, CAL_BUFFER)

    # 1) 加载 OHLCV window
    ohlcv = load_ohlcv_window(ohlcv_files, year, date_list)

    # 2) pivot 成矩阵：calendar x stock
    instruments = ohlcv["instrument_norm"].astype(str).unique()
    instruments.sort()

    close_df = ohlcv.pivot(index="date", columns="instrument_norm", values="close").reindex(date_list)
    vol_df = ohlcv.pivot(index="date", columns="instrument_norm", values="volume").reindex(date_list)
    close_df = close_df.reindex(columns=instruments)
    vol_df = vol_df.reindex(columns=instruments)

    close_mat = close_df.to_numpy(dtype=np.float32)
    vol_mat = vol_df.to_numpy(dtype=np.float32)

    # 3) 计算 label_ret_1d（关键改动：若 t+1 停牌，则 t 的 ret label 无效）
    present = ~np.isnan(close_mat)
    vol0 = np.where(np.isnan(vol_mat), 0.0, vol_mat)
    is_trade = present & (vol0 > 0)                 # 当天是否可交易（volume>0）
    is_trade_next = np.roll(is_trade, shift=-1, axis=0)
    is_trade_next[-1, :] = False                    # 最后一行没有 t+1
    close_next = np.roll(close_mat, shift=-1, axis=0)
    close_next[-1, :] = np.nan

    # 先算收益
    ret_mat = (close_next - close_mat) / close_mat
    # 再按照可交易约束做 mask：t、t+1 都必须可交易，否则置 NaN
    ret_valid_mat = is_trade & is_trade_next
    ret_mat[~ret_valid_mat] = np.nan

    # 4) 计算动量分类矩阵
    mom_mat = compute_momentum_matrix(date_list, close_mat, vol_mat, GAP_L, LINE_LEN)

    date_index = pd.Index(date_list)
    inst_index = pd.Index(instruments)

    # 5) 逐 row-group 读取 feature，生成 norm 列并打标签写出
    pf = pq.ParquetFile(feature_path)
    fcols = pf.schema_arrow.names

    f_dcol = pick_col(fcols, ["date", "trade_date", "datetime"], name="feature 日期列")
    f_icol = pick_col(fcols, ["instrument", "order_book_id", "code", "symbol", "ts_code"], name="feature 股票列")

    writer = None
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()

        # 不动原列，新增 norm 列用于匹配
        df["date_norm"] = normalize_date_series(df[f_dcol])
        df["instrument_norm"] = normalize_instrument_series(df[f_icol])

        # indexer：缺失填充只是为了避免 get_indexer 报错，最终靠 valid mask 控制
        di = date_index.get_indexer(df["date_norm"].fillna(pd.Timestamp("1900-01-01")).values)
        ii = inst_index.get_indexer(df["instrument_norm"].fillna("__MISSING__").values)

        valid = (
            (di >= 0) & (ii >= 0) &
            df["date_norm"].notna().to_numpy() &
            df["instrument_norm"].notna().to_numpy()
        )

        # 先全部置缺失
        label_ret = np.full(len(df), np.nan, dtype=np.float32)
        label_mom = np.full(len(df), -1, dtype=np.int16)

        # 填入有效位置
        label_ret[valid] = ret_mat[di[valid], ii[valid]].astype(np.float32)
        label_mom[valid] = mom_mat[di[valid], ii[valid]].astype(np.int16)

        # 写入列
        df["label_ret_1d"] = label_ret
        df["label_mom_cls"] = label_mom.astype(np.int16)

        # mask 列（LSTM 训练用：不删行，用 mask 控制 loss）
        df["label_ret_valid"] = ~np.isnan(label_ret)       # 注意：t+1 停牌会导致这里 False
        df["label_mom_valid"] = (label_mom != -1)
        df["label_both_valid"] = df["label_ret_valid"] & df["label_mom_valid"]

        # 仅按年份过滤（不按 label 过滤）
        # 如果你的 yearly parquet 本来就是该年的数据，这一步不会删任何有效行
        df = df[df["date_norm"].isin(year_dates)]

        out_table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, out_table.schema, compression="zstd")
        writer.write_table(out_table)

    if writer is not None:
        writer.close()


# ============================================================
# 输出后检查：缺失率 + 0率
# ============================================================
def check_missing_and_zero_rates(labeled_dir: str, out_csv: str, out_json: str):
    """
    逐年统计：
    - ret_missing_rate / mom_missing_rate / both_missing_rate
    - ret_zero_rate_among_valid（只在 label_ret_valid==True 的样本里统计 ret==0）
    """
    files = find_year_parquet_files(labeled_dir, prefer_prefix="year=")
    if not files:
        raise ValueError(f"未找到 labeled parquet：{labeled_dir}")

    rows = []
    for path in files:
        y = _extract_year_from_name(os.path.basename(path))
        pf = pq.ParquetFile(path)
        cols = set(pf.schema_arrow.names)

        need = ["label_ret_1d", "label_mom_cls", "label_ret_valid", "label_mom_valid", "label_both_valid"]
        for c in need:
            if c not in cols:
                raise ValueError(f"[{os.path.basename(path)}] 缺少列 {c}")

        total = 0
        ret_valid_cnt = 0
        mom_valid_cnt = 0
        both_valid_cnt = 0
        ret_zero_cnt = 0

        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=need).to_pandas()
            n = len(t)
            total += n

            rv = t["label_ret_valid"].astype(bool).to_numpy()
            mv = t["label_mom_valid"].astype(bool).to_numpy()
            bv = t["label_both_valid"].astype(bool).to_numpy()

            ret_valid_cnt += int(rv.sum())
            mom_valid_cnt += int(mv.sum())
            both_valid_cnt += int(bv.sum())

            # 只在 valid 里统计 0
            ret = pd.to_numeric(t["label_ret_1d"], errors="coerce").to_numpy()
            ret_zero_cnt += int(np.sum(rv & (np.abs(ret) <= ZERO_EPS)))

        ret_missing_cnt = total - ret_valid_cnt
        mom_missing_cnt = total - mom_valid_cnt
        both_missing_cnt = total - both_valid_cnt

        rows.append({
            "year": y,
            "rows": total,
            "ret_missing_rate": (ret_missing_cnt / total) if total else np.nan,
            "mom_missing_rate": (mom_missing_cnt / total) if total else np.nan,
            "both_missing_rate": (both_missing_cnt / total) if total else np.nan,
            "ret_zero_rate_among_valid": (ret_zero_cnt / ret_valid_cnt) if ret_valid_cnt else np.nan,
            "ret_valid_cnt": ret_valid_cnt,
            "mom_valid_cnt": mom_valid_cnt,
            "both_valid_cnt": both_valid_cnt,
        })

    report = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return report


# ============================================================
# 动量分类审计：抽样复算一致性（独立于输出 mom_mat）
# ============================================================
def reservoir_sample_mom_valid(pf: pq.ParquetFile, k: int, rng: np.random.Generator):
    """
    在不读全表的情况下，从 pf 中对 label_mom_valid==True 的行做水库抽样，返回样本列表：
    [(date_norm, instrument_norm, stored_label), ...]
    """
    samples = []
    seen = 0

    cols = set(pf.schema_arrow.names)
    need = []
    # 优先使用 date_norm/instrument_norm（脚本输出里有）
    if "date_norm" in cols:
        need.append("date_norm")
    else:
        need.append(pick_col(cols, ["date", "trade_date", "datetime"], name="date"))

    if "instrument_norm" in cols:
        need.append("instrument_norm")
    else:
        need.append(pick_col(cols, ["instrument", "order_book_id", "code", "symbol", "ts_code"], name="instrument"))

    need += ["label_mom_cls", "label_mom_valid"]

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=need).to_pandas()

        # 统一列名
        if "date_norm" not in df.columns:
            df["date_norm"] = normalize_date_series(df[need[0]])
        else:
            df["date_norm"] = normalize_date_series(df["date_norm"])

        if "instrument_norm" not in df.columns:
            df["instrument_norm"] = normalize_instrument_series(df[need[1]])
        else:
            df["instrument_norm"] = df["instrument_norm"].astype(str)

        mv = df["label_mom_valid"].astype(bool).to_numpy()
        if mv.sum() == 0:
            continue

        sub = df.loc[mv, ["date_norm", "instrument_norm", "label_mom_cls"]].dropna()
        if sub.empty:
            continue

        for _, r in sub.iterrows():
            seen += 1
            item = (pd.Timestamp(r["date_norm"]), str(r["instrument_norm"]), int(r["label_mom_cls"]))
            if len(samples) < k:
                samples.append(item)
            else:
                j = rng.integers(0, seen)
                if j < k:
                    samples[j] = item

    return samples


def recompute_one_mom_label_from_matrices(close_col: np.ndarray, vol_col: np.ndarray, p: int,
                                         gap_l: int, line_len: int) -> int:
    """
    对单只股票在 date_list[p] 位置复算动量分类（与 compute_momentum_matrix 的规则一致）
    返回 [-1,0..4]
    """
    close = close_col
    vol0 = np.where(np.isnan(vol_col), 0.0, vol_col)

    present = ~np.isnan(close)
    is_trade = present & (vol0 > 0)

    if not present[p]:
        return -1

    missing = ~present
    last_missing = np.maximum.accumulate(np.where(missing, np.arange(len(close)), -1))

    trade_pos = np.flatnonzero(is_trade)
    if trade_pos.size == 0:
        return -1

    trade_count = np.cumsum(is_trade)
    trade_rank = np.maximum.accumulate(np.where(is_trade, trade_count - 1, -1))
    r = trade_rank[p]
    if r < 0:
        return -1

    if r - (line_len - 1) < 0:
        return -1
    if (r - (line_len - 1) - gap_l) < 0:
        return -1

    ranks = np.arange(r - (line_len - 1), r + 1, dtype=np.int32)
    t_list = trade_pos[ranks]

    m_line = np.empty(line_len, dtype=np.float32)
    for k, rr in enumerate(ranks):
        base_rank = rr - gap_l
        if base_rank < 0:
            return -1
        t0 = trade_pos[base_rank]
        if last_missing[t_list[k]] > t0:
            return -1
        m_line[k] = close[t_list[k]] - close[t0]

    return classify_momentum_line(m_line)


def audit_momentum_labels(labeled_dir: str, cal_index: pd.Index, ohlcv_files: list,
                          sample_per_year: int, seed: int,
                          out_csv: str):
    """
    对每年 labeled 文件抽样 sample_per_year 条 mom_valid 样本，
    用 OHLCV 重新 pivot + 单点复算动量分类，检查与 stored label 是否一致。
    """
    rng = np.random.default_rng(seed)
    files = find_year_parquet_files(labeled_dir, prefer_prefix="year=")
    if not files:
        raise ValueError(f"未找到 labeled parquet：{labeled_dir}")

    mismatches = []
    total_checked = 0
    total_mismatch = 0

    for path in files:
        year = _extract_year_from_name(os.path.basename(path))
        pf = pq.ParquetFile(path)

        # 抽样
        samples = reservoir_sample_mom_valid(pf, sample_per_year, rng)
        if not samples:
            continue

        # 构造 year 的 date_list，并重新加载 OHLCV window（保证复算基于原始 OHLCV）
        year_dates, date_list = build_date_window(cal_index, year, CAL_BUFFER)
        ohlcv = load_ohlcv_window(ohlcv_files, year, date_list)

        instruments = ohlcv["instrument_norm"].astype(str).unique()
        instruments.sort()

        close_df = ohlcv.pivot(index="date", columns="instrument_norm", values="close").reindex(date_list)
        vol_df = ohlcv.pivot(index="date", columns="instrument_norm", values="volume").reindex(date_list)
        close_df = close_df.reindex(columns=instruments)
        vol_df = vol_df.reindex(columns=instruments)

        close_mat = close_df.to_numpy(dtype=np.float32)
        vol_mat = vol_df.to_numpy(dtype=np.float32)

        date_index = pd.Index(date_list)
        inst_index = pd.Index(instruments)

        # 对每条样本复算
        for d, inst, stored in samples:
            if d not in date_index:
                total_checked += 1
                total_mismatch += 1
                mismatches.append({
                    "year": year, "instrument_norm": inst, "date": str(d.date()),
                    "stored": stored, "recomputed": -999, "reason": "date_not_in_window"
                })
                continue
            if inst not in inst_index:
                total_checked += 1
                total_mismatch += 1
                mismatches.append({
                    "year": year, "instrument_norm": inst, "date": str(d.date()),
                    "stored": stored, "recomputed": -999, "reason": "instrument_not_in_ohlcv_window"
                })
                continue

            p = date_index.get_loc(d)
            j = inst_index.get_loc(inst)

            recomputed = recompute_one_mom_label_from_matrices(close_mat[:, j], vol_mat[:, j], p, GAP_L, LINE_LEN)

            total_checked += 1
            if stored != recomputed:
                total_mismatch += 1
                mismatches.append({
                    "year": year, "instrument_norm": inst, "date": str(d.date()),
                    "stored": stored, "recomputed": int(recomputed), "reason": "label_mismatch"
                })

    mismatch_rate = (total_mismatch / total_checked) if total_checked else 0.0
    dfm = pd.DataFrame(mismatches)
    dfm.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n[动量分类审计]")
    print(f"checked={total_checked}, mismatches={total_mismatch}, mismatch_rate={mismatch_rate:.6f}")
    print(f"mismatch list saved: {out_csv}")
    if not dfm.empty:
        print("Top mismatches (first 20):")
        print(dfm.head(20).to_string(index=False))


# ============================================================
# 主程序
# ============================================================
def main():
    # 交易日历
    cal_index = load_trading_calendar(CAL_PATH)

    # 扫描 feature 年文件
    feature_files = find_year_parquet_files(FEATURE_DIR, prefer_prefix="year=")
    if not feature_files:
        raise ValueError(f"FEATURE_DIR 下找不到任何 yearly parquet：{FEATURE_DIR}")

    years = sorted({_extract_year_from_name(os.path.basename(p)) for p in feature_files})
    print(f"Found feature years: {years[0]} ... {years[-1]}  (total {len(years)})")
    print(f"Momentum params: GAP_L={GAP_L}, LINE_LEN={LINE_LEN}, CAL_BUFFER={CAL_BUFFER}")
    print(f"Output dir: {OUT_DIR}")

    # 扫描 OHLCV 年文件
    ohlcv_files = find_year_parquet_files(OHLCV_DIR, prefer_prefix="ohlcv_")
    if not ohlcv_files:
        raise ValueError(f"OHLCV_DIR 下找不到任何 yearly parquet：{OHLCV_DIR}")

    # 逐年打标签
    for y in years:
        in_path = pick_best_year_file(
            feature_files, y,
            allow_patterns=[r"^year=", r"^\d{4}\.parquet$"]
        )
        if in_path is None:
            print(f"[WARN] year {y} feature file not found, skip.")
            continue

        out_path = os.path.join(OUT_DIR, f"year={y}.parquet")
        print(f"\n=== Labeling year {y}: {in_path} -> {out_path}")
        write_labeled_year(in_path, out_path, cal_index, ohlcv_files, y)

    # 写 label 映射说明
    mapping_path = os.path.join(OUT_DIR, "_label_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("label_mom_cls encoding:\n")
        f.write("4 = Bounce (neg->pos)\n")
        f.write("3 = Positive (all > 0)\n")
        f.write("2 = Volatile (oscillate / multi-cross)\n")
        f.write("1 = Negative (all < 0)\n")
        f.write("0 = Sink (pos->neg)\n")
        f.write("\nMissing/invalid momentum label is -1.\n")
        f.write("label_ret_1d missing/invalid is NaN.\n")
        f.write("Important: ret label requires BOTH t and t+1 to be tradeable (volume>0).\n")

    # 检查：缺失率 + 0率
    report_csv = os.path.join(OUT_DIR, "label_missing_and_zero_rates_by_year.csv")
    report_json = os.path.join(OUT_DIR, "label_missing_and_zero_rates_by_year.json")
    rep = check_missing_and_zero_rates(OUT_DIR, report_csv, report_json)

    print("\n[报告] 每年缺失率 + 0率（仅统计 valid 的 ret=0）")
    print(rep)

    # 审计：动量分类抽样复算一致性
    audit_csv = os.path.join(OUT_DIR, "audit_momentum_mismatches.csv")
    audit_momentum_labels(
        labeled_dir=OUT_DIR,
        cal_index=cal_index,
        ohlcv_files=ohlcv_files,
        sample_per_year=AUDIT_MOM_SAMPLE_PER_YEAR,
        seed=AUDIT_SEED,
        out_csv=audit_csv
    )

    print("\nDone.")
    print("Saved mapping:", mapping_path)
    print("Saved report:", report_csv)
    print("Saved report:", report_json)
    print("Saved momentum audit:", audit_csv)


if __name__ == "__main__":
    main()
