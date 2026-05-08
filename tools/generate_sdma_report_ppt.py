#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import uno  # type: ignore
from com.sun.star.awt import Point, Size  # type: ignore
from com.sun.star.beans import PropertyValue  # type: ignore


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PPTX = OUTPUT_DIR / "sdma_prefetch_report.pptx"
OUTPUT_ODP = OUTPUT_DIR / "sdma_prefetch_report.odp"


SLIDE_W = 28000
SLIDE_H = 15750


def make_prop(name: str, value):
    prop = PropertyValue()
    prop.Name = name
    prop.Value = value
    return prop


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def ensure_office(host: str = "127.0.0.1", port: int = 2002) -> subprocess.Popen | None:
    if is_port_open(host, port):
        return None
    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        f'--accept=socket,host={host},port={port};urp;StarOffice.ServiceManager',
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if is_port_open(host, port):
            return proc
        time.sleep(0.2)
    raise RuntimeError("Failed to start LibreOffice headless listener")


def connect_office(host: str = "127.0.0.1", port: int = 2002):
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    ctx = resolver.resolve(
        f"uno:socket,host={host},port={port};urp;StarOffice.ComponentContext"
    )
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    return ctx, smgr, desktop


def add_shape(doc, page, kind: str, x: int, y: int, w: int, h: int):
    shape = doc.createInstance(kind)
    shape.setPosition(Point(X=x, Y=y))
    shape.setSize(Size(Width=w, Height=h))
    page.add(shape)
    return shape


def add_textbox(doc, page, x, y, w, h, text, *, font_size=20, bold=False,
                fill=None, line=None, color=0x111111):
    shape = add_shape(doc, page, "com.sun.star.drawing.TextShape", x, y, w, h)
    if fill is not None:
        shape.FillStyle = 1
        shape.FillColor = fill
    else:
        shape.FillStyle = 0
    if line is not None:
        shape.LineColor = line
    else:
        shape.LineStyle = 0
    shape.TextAutoGrowHeight = True
    shape.TextLeftDistance = 250
    shape.TextRightDistance = 250
    shape.TextUpperDistance = 180
    shape.TextLowerDistance = 180
    shape.String = text
    cursor = shape.createTextCursor()
    cursor.gotoStart(False)
    cursor.gotoEnd(True)
    cursor.CharHeight = font_size
    cursor.CharColor = color
    cursor.CharWeight = 150 if bold else 100
    return shape


def add_title(doc, page, title: str, subtitle: str | None = None):
    add_textbox(doc, page, 900, 500, 26200, 1700, title,
                font_size=28, bold=True, color=0x0B2340)
    if subtitle:
        add_textbox(doc, page, 920, 2200, 25000, 900, subtitle,
                    font_size=14, color=0x4A5568)


def add_bullets(doc, page, lines: list[str], *, x=1200, y=3200, w=25000, h=10500,
                font_size=18, fill=None):
    text = "\n".join(f"• {line}" for line in lines)
    add_textbox(doc, page, x, y, w, h, text, font_size=font_size, fill=fill)


def add_flow_boxes(doc, page):
    boxes = [
        ("Benchmark\nbenchmark_rounds / avg_aicore", 1200, 4100, 3600, 1800, 0xDCEEFF),
        ("Python入口\nrun_example / CodeRunner", 5900, 4100, 3600, 1800, 0xE8F5E9),
        ("Host侧setup\nprovider + stream + workspace", 10600, 4100, 4200, 1800, 0xFFF3D6),
        ("AICPU调度\neligibility + issue", 15850, 4100, 3600, 1800, 0xFDE7E9),
        ("STARS SDMA\n写SQE + doorbell", 20450, 4100, 3600, 1800, 0xEDE7F6),
    ]
    for text, x, y, w, h, fill in boxes:
        add_textbox(doc, page, x, y, w, h, text, font_size=16, bold=True, fill=fill, line=0xA0AEC0)
    arrows = ["→", "→", "→", "→"]
    xs = [5000, 9850, 15100, 19950]
    for arrow, x in zip(arrows, xs):
        add_textbox(doc, page, x, 4550, 600, 600, arrow, font_size=26, bold=True, color=0x2D3748)


def create_slide(doc, pages, index: int):
    if index < pages.getCount():
        return pages.getByIndex(index)
    pages.insertNewByIndex(index)
    return pages.getByIndex(index)


def build_presentation(doc):
    pages = doc.getDrawPages()

    # Slide 1
    page = create_slide(doc, pages, 0)
    add_title(doc, page, "SDMA 预取方案汇报", "a2a3 / tensormap_and_ringbuffer / benchmark_rounds_v1 结果")
    add_bullets(doc, page, [
        "问题背景：baseline 路径无法提前把关键数据 warm 到 AICore L2，部分 workload 的 device 端 E2E 仍有优化空间。",
        "方案目标：在不改业务输入和调度主干的前提下，引入 AICPU 发起的 STARS SDMA prefetch。",
        "汇报范围：背景、整体设计思路、关键阈值、benchmark 方法、30 轮实测结果和结论。",
    ], y=3600, font_size=20)

    # Slide 2
    page = create_slide(doc, pages, 1)
    add_title(doc, page, "1. 问题背景")
    add_bullets(doc, page, [
        "当前 compare 模式是 baseline 与 sdma 两次独立运行，workload 和 case 保持一致，只比较 prefetch 路径差异。",
        "baseline：不建 SDMA workspace，不发 prefetch，AICore 直接等真实调度触发。",
        "sdma：host 预建 STARS channel，AICPU 在 dispatch 邻近点异步发起 prefetch。",
        "核心判断指标不是单轮 profiling，而是多轮 device log E2E 平均值，因为后者更稳定、更接近最终收益。",
    ])

    # Slide 3
    page = create_slide(doc, pages, 2)
    add_title(doc, page, "2. SDMA 模式整体设计思路")
    add_bullets(doc, page, [
        "设计原则 1：预取是可选增强，不是主流程依赖；provider 或 channel 缺失时自动降级为 no-op。",
        "设计原则 2：把地址和字节数推导前移到 payload 提交阶段，调度热路径只做轻量 eligibility 判定。",
        "设计原则 3：只在当前任务首个 block dispatch 后，对 batch 内下一个候选任务做 look-ahead 预取。",
        "设计原则 4：通过 min_bytes 和 suppression window 控制激进度，避免控制开销吞掉收益。",
    ])

    # Slide 4
    page = create_slide(doc, pages, 3)
    add_title(doc, page, "3. 端到端路径")
    add_flow_boxes(doc, page)
    add_bullets(doc, page, [
        "host 侧：准备 provider、device-only stream、workspace，并把 STARS channel info 写入 Runtime。",
        "runtime 侧：解析 prefetch_mode / min_bytes / suppress_window / debug 等配置。",
        "AICPU 侧：判断 payload 是否满足预取资格，再调用 aicpu_prefetch_issue_reserved 写 SQE。",
        "device 侧：向 STARS SQ 写入 CMO PREFETCH SQE，最后只敲一次 doorbell。",
    ], y=7000, h=5000, font_size=17)

    # Slide 5
    page = create_slide(doc, pages, 4)
    add_title(doc, page, "4. 当前关键阈值与规则")
    add_bullets(doc, page, [
        "PTO_SDMA_PREFETCH_MIN_BYTES = 256 KB：用于 eligibility filter，比的是 prefetch_filter_bytes，不是单次 issue_bytes。",
        "PTO_SDMA_PREFETCH_SUPPRESS_WINDOW 默认 = 2：成功 issue 后，同一 channel 后续 eligible attempt 先跳过。",
        "workload-specific suppression：Case2-like = 7，Case1-like = 5，batch_paged_attention = 7，generic = 31。",
        "典型 issue_bytes：paged_attention_unroll Case2-like ≈ 16 KB，Case1-like ≈ 32 KB+。",
        "含义：当前实现是‘保守但稳定’，目标是避免小任务高频小额预取。",
    ])

    # Slide 6
    page = create_slide(doc, pages, 5)
    add_title(doc, page, "5. Benchmark 方法")
    add_bullets(doc, page, [
        "命令：./tools/benchmark_rounds_v1.sh -p a2a3 -r tensormap_and_ringbuffer --rounds 30",
        "profiling 三列：单独跑 1 次 profiling run，关注 AICore Exec / AICPU Dispatch->Finish / Device E2E(profiling)。",
        "device log 一列：单独跑 30 轮 non-profiling run，对 30 轮 Device E2E 求平均。",
        "判断收益时优先看 Device E2E Avg(device log)，因为单轮 profiling 波动更大。",
    ])

    # Slide 7
    page = create_slide(doc, pages, 6)
    add_title(doc, page, "6. 结果汇总")
    table_text = (
        "Workload                          Base(us)   SDMA(us)   Gain(us)   Gain(%)\n"
        "alternating_matmul_add             965.17     946.23      18.94      1.96%\n"
        "benchmark_bgemm                    790.10     759.38      30.72      3.89%\n"
        "paged_attention_unroll Case1      1204.11    1193.19      10.92      0.91%\n"
        "paged_attention_unroll Case2       604.85     594.23      10.62      1.76%\n"
        "batch_paged_attention             3546.56    3493.29      53.27      1.50%\n"
        "\n"
        "简单平均收益：约 2.00%"
    )
    add_textbox(doc, page, 1200, 3200, 24800, 9000, table_text,
                font_size=16, fill=0xF7FAFC, line=0xCBD5E0)

    # Slide 8
    page = create_slide(doc, pages, 7)
    add_title(doc, page, "7. 结果解读")
    add_bullets(doc, page, [
        "5 个 workload 的 device-log 平均值全部正收益，没有出现回退。",
        "相对收益最高的是 benchmark_bgemm：+3.89%，说明规则访问、较大粒度数据路径最受益。",
        "绝对收益最大的是 batch_paged_attention：节省约 53.27 us，说明长尾 workload 的累计收益更可观。",
        "paged_attention_unroll Case1 / Case2 也有正收益，但幅度更保守，说明 16 KB / 32 KB+ 两档参数还有细化空间。",
    ])

    # Slide 9
    page = create_slide(doc, pages, 8)
    add_title(doc, page, "8. 结论")
    add_bullets(doc, page, [
        "结论 1：当前 SDMA prefetch 已从‘功能可用’进入‘有稳定小幅收益’阶段。",
        "结论 2：这组数据下，device-log 平均收益区间为 0.91% ~ 3.89%，简单平均约 2.00%。",
        "结论 3：更适合用多轮 Device E2E Avg(device log) 评估收益，而不是只看单轮 profiling。",
        "结论 4：下一步重点不是证明‘有没有收益’，而是做 workload-aware 参数细化，把当前 ~2% 继续往上推。",
    ])

    # Slide 10
    page = create_slide(doc, pages, 9)
    add_title(doc, page, "9. 下一步建议")
    add_bullets(doc, page, [
        "对 benchmark_bgemm 与 batch_paged_attention 继续放大样本，验证收益是否稳定复现。",
        "按 issue_bytes 分档细化 min_bytes 与 suppression window，而不是继续用固定 5/7/31。",
        "继续完善结果归档，把 mode / rounds / profiling 状态和输出文件一一对应。",
        "如需继续优化 device 侧行为，可评估一次提交双 SQE（数据预取 + 指令预取）的实现收益。",
    ])


def save_doc(doc, path: Path, filter_name: str):
    props = (
        make_prop("FilterName", filter_name),
        make_prop("Overwrite", True),
    )
    doc.storeAsURL(uno.systemPathToFileUrl(str(path)), props)


def main():
    soffice_proc = None
    desktop = None
    try:
        soffice_proc = ensure_office()
        _, _, desktop = connect_office()
        doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        build_presentation(doc)
        save_doc(doc, OUTPUT_ODP, "impress8")
        save_doc(doc, OUTPUT_PPTX, "Impress MS PowerPoint 2007 XML")
        doc.close(True)
        print(OUTPUT_PPTX)
    finally:
        if soffice_proc is not None and desktop is not None:
            try:
                desktop.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
