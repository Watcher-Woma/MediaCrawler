# -*- coding: utf-8 -*-
"""
小红书「AI科研」搜索结果数据可视化脚本
=====================================
数据来源（同目录 jsonl）：
  - search_contents_2026-10-26.jsonl : 笔记（帖子）元数据，80 条
  - search_comments_2026-10-26.jsonl : 评论数据，749 条

本脚本自动完成：
  1. 读取并清洗两类数据（含「万」单位的数值解析）
  2. 计算多种聚合指标
  3. 生成 8 张主题图表（构成 / 对比 / 排名 / 分布 / 时间序列 / 相关性）
  4. 图表均包含标题、坐标轴标签、图例等必要元素
  5. 输出 PNG 至 output_xhs_viz/ 目录

运行方式（隔离 venv，已安装 pandas/matplotlib/numpy）：
  python xhs_visualize.py
"""

import os
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端，便于在服务器/脚本环境生成图片
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ----------------------------------------------------------------------------
# 0. 中文字体配置（Windows 自带黑体，保证中文正常显示）
# ----------------------------------------------------------------------------
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
plt.rcParams["figure.dpi"] = 130            # 提高出图清晰度
plt.rcParams["savefig.bbox"] = "tight"

# 统一配色（专业、对比清晰）
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
ACCENT = "#2F5597"

# ----------------------------------------------------------------------------
# 1. 数据加载与清洗
# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CONTENT_FILE = os.path.join(BASE, "data", "xhs", "jsonl", "search_contents_2026-10-26.jsonl")
COMMENT_FILE = os.path.join(BASE, "data", "xhs", "jsonl", "search_comments_2026-10-26.jsonl")
OUT_DIR = os.path.join(BASE, "output_xhs_viz")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(s):
    """去除标题中的 emoji 及符号区字符（如 ⭐），避免中文字体缺字。"""
    if not isinstance(s, str):
        return ""
    out = []
    for ch in s:
        o = ord(ch)
        if o <= 0xFFFF and not (0x2600 <= o <= 0x27BF) and not (0x2B00 <= o <= 0x2BFF):
            out.append(ch)
    return "".join(out)


def parse_num(s):
    """把 '1.3万' / '4666' / '' 统一解析为 float，无法解析返回 NaN。"""
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    s = s.replace(",", "")
    if "万" in s:
        return float(s.replace("万", "")) * 10000
    if "w" in s.lower():
        return float(s.lower().replace("w", "")) * 10000
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


content_rows = load_jsonl(CONTENT_FILE)
comment_rows = load_jsonl(COMMENT_FILE)

# 笔记表
df_c = pd.DataFrame(content_rows)
for col in ["liked_count", "collected_count", "comment_count", "share_count"]:
    df_c[col + "_num"] = df_c[col].map(parse_num)
df_c["time_dt"] = pd.to_datetime(df_c["time"], unit="ms")
df_c["type_label"] = df_c["type"].map({"normal": "图文", "video": "视频"}).fillna(df_c["type"])

# 评论表
df_m = pd.DataFrame(comment_rows)
df_m["like_count_num"] = df_m["like_count"].map(parse_num)
df_m["sub_comment_count_num"] = df_m["sub_comment_count"].map(parse_num)
df_m["create_dt"] = pd.to_datetime(df_m["create_time"], unit="ms")

print(f"笔记数: {len(df_c)}  评论数: {len(df_m)}")

# ----------------------------------------------------------------------------
# 2. 各图表生成
# ----------------------------------------------------------------------------

# ---- 图1：笔记类型构成（环形图）---------------------------------------------
def chart_type_donut():
    counts = df_c["type_label"].value_counts()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct=lambda p: f"{p:.1f}%\n({int(round(p*counts.sum()/100))})",
        colors=PALETTE, startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
        pctdistance=0.78, textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontweight("bold")
    ax.set_title("笔记类型构成（图文 vs 视频）", fontsize=14, fontweight="bold")
    ax.text(0, 0, f"共\n{len(df_c)}篇", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.legend(wedges, [f"{l}（{n}）" for l, n in counts.items()],
              title="类型", loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.savefig(os.path.join(OUT_DIR, "01_笔记类型构成.png"))
    plt.close(fig)


# ---- 图2：各搜索关键词平均互动指标（分组柱状图）-----------------------------
def chart_keyword_grouped():
    kw = df_c.groupby("source_keyword")[["liked_count_num", "collected_count_num",
                                          "comment_count_num", "share_count_num"]].mean() / 10000
    metrics = ["liked_count_num", "collected_count_num", "comment_count_num", "share_count_num"]
    labels = ["平均点赞", "平均收藏", "平均评论", "平均分享"]
    x = np.arange(len(kw.index))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * width, kw[m], width, label=lab, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(kw.index, fontsize=11)
    ax.set_ylabel("数值（万次）", fontsize=11)
    ax.set_xlabel("搜索关键词", fontsize=11)
    ax.set_title("各搜索关键词下笔记的平均互动指标对比", fontsize=14, fontweight="bold")
    ax.legend(title="互动指标", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "02_关键词平均互动对比.png"))
    plt.close(fig)


# ---- 图3：Top10 高赞笔记（水平条形图）---------------------------------------
def chart_top_notes():
    top = df_c.sort_values("liked_count_num", ascending=False).head(10).copy()
    top["short_title"] = top["title"].map(clean_text).str.slice(0, 16).fillna("") + "…"
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(top))[::-1]
    bars = ax.barh(y, top["liked_count_num"], color=ACCENT)
    ax.set_yticks(y)
    ax.set_yticklabels(top["short_title"], fontsize=10)
    ax.set_xlabel("点赞数", fontsize=11)
    ax.set_title("点赞数最高的 Top 10 笔记", fontsize=14, fontweight="bold")
    for b, v in zip(bars, top["liked_count_num"]):
        ax.text(b.get_width() + max(top["liked_count_num"]) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v/10000:.1f}万" if v >= 10000 else f"{int(v)}", va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "03_Top10高赞笔记.png"))
    plt.close(fig)


# ---- 图4：笔记四项互动指标分布（箱线图）-------------------------------------
def chart_box_engagement():
    cols = {"liked_count_num": "点赞", "collected_count_num": "收藏",
            "comment_count_num": "评论", "share_count_num": "分享"}
    data = [df_c[c].dropna() for c in cols]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bp = ax.boxplot(data, tick_labels=list(cols.values()), patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    showfliers=True, flierprops=dict(marker="o", markersize=4, alpha=0.4))
    for patch, color in zip(bp["boxes"], PALETTE[:4]):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    ax.set_ylabel("数值（次）", fontsize=11)
    ax.set_title("笔记四项互动指标分布（箱线图，含离群点）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "04_互动指标箱线图.png"))
    plt.close(fig)


# ---- 图5：评论点赞数分布（直方图）-------------------------------------------
def chart_comment_like_hist():
    likes = df_m["like_count_num"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # 用对数分箱，因点赞数跨度极大（0 ~ 29000）
    bins = np.logspace(0, np.log10(likes.max() + 1), 30)
    ax.hist(likes, bins=bins, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("评论点赞数（对数刻度）", fontsize=11)
    ax.set_ylabel("评论条数", fontsize=11)
    ax.set_title("评论点赞数分布（对数刻度直方图）", fontsize=14, fontweight="bold")
    ax.axvline(likes.median(), color=PALETTE[3], linestyle="--", linewidth=1.5,
               label=f"中位数={likes.median():.0f}")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "05_评论点赞分布.png"))
    plt.close(fig)


# ---- 图6：评论数最多的 Top15 笔记（水平条形图）------------------------------
def chart_top_commented():
    cpm = df_m.groupby("note_id").size().reset_index(name="comments")
    merged = cpm.merge(df_c[["note_id", "title"]], on="note_id", how="left")
    merged["short_title"] = merged["title"].map(clean_text).str.slice(0, 16).fillna("") + "…"
    top = merged.sort_values("comments", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(top))[::-1]
    bars = ax.barh(y, top["comments"], color=PALETTE[2])
    ax.set_yticks(y)
    ax.set_yticklabels(top["short_title"], fontsize=9)
    ax.set_xlabel("评论数（条）", fontsize=11)
    ax.set_title("评论数最多的 Top 15 笔记", fontsize=14, fontweight="bold")
    for b, v in zip(bars, top["comments"]):
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, str(int(v)), va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "06_Top15评论数笔记.png"))
    plt.close(fig)


# ---- 图7：评论时间趋势（按月的折线/面积图）----------------------------------
def chart_comment_trend():
    s = df_m.set_index("create_dt").resample("MS").size()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.fill_between(s.index, s.values, color=PALETTE[0], alpha=0.25)
    ax.plot(s.index, s.values, color=PALETTE[0], marker="o", linewidth=2, label="月度评论数")
    ax.set_xlabel("月份", fontsize=11)
    ax.set_ylabel("评论数（条）", fontsize=11)
    ax.set_title("评论发布时间趋势（按月聚合）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.35)
    fig.autofmt_xdate()
    fig.savefig(os.path.join(OUT_DIR, "07_评论时间趋势.png"))
    plt.close(fig)


# ---- 图8：笔记点赞 vs 评论 相关性（散点图 + 拟合线）--------------------------
def chart_scatter_corr():
    sub = df_c.dropna(subset=["liked_count_num", "comment_count_num"])
    x = sub["liked_count_num"].values
    y = sub["comment_count_num"].values
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = [PALETTE[0] if t == "图文" else PALETTE[1] for t in sub["type_label"]]
    ax.scatter(x, y, c=colors, s=45, alpha=0.75, edgecolor="white", linewidth=0.5)
    # 线性拟合
    if len(x) > 1:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, color=PALETTE[3], linewidth=1.8,
                label=f"线性拟合 (r≈{np.corrcoef(x, y)[0,1]:.2f})")
    ax.set_xlabel("点赞数", fontsize=11)
    ax.set_ylabel("评论数", fontsize=11)
    ax.set_title("笔记点赞数与评论数关系（散点图）", fontsize=14, fontweight="bold")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[0], markersize=9, label="图文"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[1], markersize=9, label="视频")]
    ax.legend(handles=legend_elems + [ax.lines[-1]], fontsize=9, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    fig.savefig(os.path.join(OUT_DIR, "08_点赞评论相关性.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. 执行全部图表
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    chart_type_donut()
    chart_keyword_grouped()
    chart_top_notes()
    chart_box_engagement()
    chart_comment_like_hist()
    chart_top_commented()
    chart_comment_trend()
    chart_scatter_corr()
    print("全部图表已生成至：", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        print("  -", f)
