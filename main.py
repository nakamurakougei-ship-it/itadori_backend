# イタドリ - 木取りエンジン + FastAPI
# 外部から REST API で木取り・木取図生成を呼び出し可能。
# 起動: uvicorn main:app --reload  (開発)  /  uvicorn main:app --host 0.0.0.0 --port 8000  (本番)

import sys
from types import ModuleType

# --- Python 3.12/3.13 互換性パッチ ---
if "distutils" not in sys.modules:
    d = ModuleType("distutils")
    d.version = ModuleType("distutils.version")

    class LooseVersion:
        def __init__(self, vstring):
            self.vstring = vstring

        def __lt__(self, other):
            return False

    d.version.LooseVersion = LooseVersion
    sys.modules["distutils"] = d
    sys.modules["distutils.version"] = d.version

import html
import io
import os

_root = os.path.dirname(os.path.abspath(__file__))


# --- 木取りエンジン ---


def _normalize_part(p):
    """長方形部品は定尺板の長手方向に長辺を沿わせるため、w=長辺・d=短辺に正規化する。"""
    w, d = p["w"], p["d"]
    return {**p, "w": max(w, d), "d": min(w, d)}


class TrunkTechEngine:
    """定尺板への木取りネスティングを行うエンジン。"""

    def __init__(self, kerf: float = 3.0):
        self.kerf = kerf

    def pack_sheets(self, parts, vw, vh):
        """
        定尺板 vw(長手) x vh(短手) に部品を配置する。
        長方形部品は必ず長辺を長手方向(vw)に、短辺を短手方向(vh)に配置する。
        定尺を超える部品は配置しない。

        parts: [{"n": 名称, "w": 幅, "d": 奥行}, ...]
        vw, vh: 定尺の長手・短手 (mm)
        戻り値: [{"id": 1, "rows": [{"y", "h", "used_w", "parts": [{"n","x","y","w","h"}, ...]}, ...]}, ...]
        """
        normalized = [_normalize_part(dict(p)) for p in parts]
        valid = [p for p in normalized if p["w"] <= vw and p["d"] <= vh]
        sorted_parts = sorted(valid, key=lambda x: (x["w"], x["d"]), reverse=True)
        sheets = []

        def pack(p):
            for s in sheets:
                for r in s["rows"]:
                    if r["h"] >= p["d"] and (vw - r["used_w"]) >= p["w"]:
                        r["parts"].append({
                            "n": p["n"], "x": r["used_w"], "y": r["y"],
                            "w": p["w"], "h": p["d"],
                        })
                        r["used_w"] += p["w"] + self.kerf
                        return True
                if (vh - s["used_h"]) >= p["d"]:
                    s["rows"].append({
                        "y": s["used_h"], "h": p["d"], "used_w": p["w"] + self.kerf,
                        "parts": [{"n": p["n"], "x": 0, "y": s["used_h"], "w": p["w"], "h": p["d"]}],
                    })
                    s["used_h"] += p["d"] + self.kerf
                    return True
            return False

        for p in sorted_parts:
            if not pack(p):
                if p["w"] <= vw and p["d"] <= vh:
                    sheets.append({
                        "id": len(sheets) + 1,
                        "used_h": p["d"] + self.kerf,
                        "rows": [{
                            "y": 0, "h": p["d"], "used_w": p["w"] + self.kerf,
                            "parts": [{"n": p["n"], "x": 0, "y": 0, "w": p["w"], "h": p["d"]}],
                        }],
                    })
        return sheets

    def pack_sheets_max(self, parts, vw, vh, max_sheets):
        """
        最大 max_sheets 枚まで詰め、収まらなかった部品を返す。
        戻り値: (sheets, unplaced_parts)
        """
        normalized = [_normalize_part(dict(p)) for p in parts]
        valid = [p for p in normalized if p["w"] <= vw and p["d"] <= vh]
        sorted_parts = sorted(valid, key=lambda x: (x["w"], x["d"]), reverse=True)
        sheets = []
        unplaced = []

        def pack(p):
            for s in sheets:
                for r in s["rows"]:
                    if r["h"] >= p["d"] and (vw - r["used_w"]) >= p["w"]:
                        r["parts"].append({
                            "n": p["n"], "x": r["used_w"], "y": r["y"],
                            "w": p["w"], "h": p["d"],
                        })
                        r["used_w"] += p["w"] + self.kerf
                        return True
                if (vh - s["used_h"]) >= p["d"]:
                    s["rows"].append({
                        "y": s["used_h"], "h": p["d"], "used_w": p["w"] + self.kerf,
                        "parts": [{"n": p["n"], "x": 0, "y": s["used_h"], "w": p["w"], "h": p["d"]}],
                    })
                    s["used_h"] += p["d"] + self.kerf
                    return True
            if len(sheets) < max_sheets and p["w"] <= vw and p["d"] <= vh:
                sheets.append({
                    "id": len(sheets) + 1,
                    "used_h": p["d"] + self.kerf,
                    "rows": [{
                        "y": 0, "h": p["d"], "used_w": p["w"] + self.kerf,
                        "parts": [{"n": p["n"], "x": 0, "y": 0, "w": p["w"], "h": p["d"]}],
                    }],
                })
                return True
            return False

        for p in sorted_parts:
            if not pack(p):
                unplaced.append(p)
        return (sheets, unplaced)


def render_sheet_to_svg(sheet, v_w_full, v_h_full, label):
    """1枚の木取図を SVG 文字列で返す（印刷用）。フォント不要で日本語も表示される。"""
    vw, vh = int(v_w_full), int(v_h_full)
    title = html.escape(f"【木取り図】 ID:{sheet['id']} ({label}：{vw}x{vh})")
    parts_svg = []
    for r in sheet["rows"]:
        for p in r["parts"]:
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            name_esc = html.escape(str(p["n"]))
            dims = f"{int(w)}×{int(h)}"
            parts_svg.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#deb887" stroke="#333" stroke-width="1"/>'
                f'<text x="{x + w/2}" y="{y + h/2}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="{min(14, h/4)}" font-weight="bold" fill="#1a1a1a">{name_esc}</text>'
                f'<text x="{x + w/2}" y="{y + h/2 + (h/4 or 10)}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="{min(10, h/6)}" fill="#444">{dims}</text>'
            )
    inner = "".join(parts_svg)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}" '
        f'style="max-width:100%;height:auto;font-family:sans-serif">'
        f'<rect x="0" y="0" width="{vw}" height="{vh}" fill="#fdf5e6" stroke="#8b4513" stroke-width="2"/>'
        f'<text x="{vw/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>'
        f'{inner}</svg>'
    )


def build_print_html(best, max_per_page=None, jp_font=None):
    """木取図を印刷用 HTML 文字列で返す（SVG 埋め込み）。シートごとに vw, vh, label があれば混在用。"""
    default_vw = best.get("vw", 0) + 2
    default_vh = best.get("vh", 0) + 2
    default_label = best.get("label", "板")
    svg_list = []
    for s in best["sheets"]:
        vw = s.get("vw", default_vw - 2) + 2
        vh = s.get("vh", default_vh - 2) + 2
        label = s.get("label", default_label)
        svg_list.append(render_sheet_to_svg(s, vw, vh, label))
    chunk = max_per_page if max_per_page is not None and max_per_page >= 1 else 1
    pages = [svg_list[i : i + chunk] for i in range(0, len(svg_list), chunk)]
    html_parts = [
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@media print { @page { size: A4; margin: 10mm; } body { margin: 0; } }
.diagram-page { page-break-after: always; padding: 0; }
.diagram-page:last-child { page-break-after: auto; }
.diagram-svg { width: 100%; max-height: 32vh; object-fit: contain; margin-bottom: 2mm; }
h1 { font-size: 14pt; margin-bottom: 4mm; }
</style></head><body>"""
    ]
    for i, page_svgs in enumerate(pages):
        html_parts.append(f'<div class="diagram-page"><h1>木取図 — {i+1}ページ目</h1>')
        for j, svg in enumerate(page_svgs):
            html_parts.append(f'<div class="diagram-svg">{svg}</div>')
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    return "".join(html_parts)


def as_long_short(a, b, lab):
    """定尺を (長手, 短手) の順で返す。板ラベル lab 付き。鼻切り -2mm は呼び出し元で適用すること。"""
    lo, sh = max(a, b), min(a, b)
    return (lo - 2, sh - 2, lab)


def find_best_mixed(parts, vw36, vh36, vw48, vh48, kerf):
    """
    3x6 のみ・4x8 のみ・4x8と3x6の混在 を試す。
    単一種で無駄（使用面積−部材面積）が少ない板種を優先し、その板種だけで済む場合は混在・他板種を選ばない。
    戻り値: (sheets, total_area)
    sheets は各枚が {"id", "label", "vw", "vh", "rows"} を持つ（混在時は枚ごとにサイズが異なる）。
    """
    engine = TrunkTechEngine(kerf=kerf)
    parts_list = [{"n": p.n, "w": p.w, "d": p.d} for p in parts]
    n_req = len(parts_list)
    parts_area = sum(p["w"] * p["d"] for p in parts_list)

    def total_area(sheets, get_vw_vh):
        a = 0
        for s in sheets:
            vw, vh = get_vw_vh(s)
            a += vw * vh
        return a

    candidates = []
    area_36_only = None
    area_48_only = None

    # 3x6 のみ
    s36 = engine.pack_sheets(parts_list, vw36, vh36)
    placed = sum(len(r["parts"]) for sh in s36 for r in sh["rows"])
    if placed == n_req:
        out = []
        for i, sh in enumerate(s36):
            out.append({"id": i + 1, "label": "3x6", "vw": vw36, "vh": vh36, "rows": sh["rows"]})
        area_36_only = total_area(out, lambda s: (s["vw"], s["vh"]))
        candidates.append((out, area_36_only))

    # 4x8 のみ
    s48 = engine.pack_sheets(parts_list, vw48, vh48)
    placed = sum(len(r["parts"]) for sh in s48 for r in sh["rows"])
    if placed == n_req:
        out = []
        for i, sh in enumerate(s48):
            out.append({"id": i + 1, "label": "4x8", "vw": vw48, "vh": vh48, "rows": sh["rows"]})
        area_48_only = total_area(out, lambda s: (s["vw"], s["vh"]))
        candidates.append((out, area_48_only))

    # 混在: 4x8 を k 枚使ってから残りを 3x6
    for k in range(1, len(s48) + 1):
        s48_k, unplaced = engine.pack_sheets_max(parts_list, vw48, vh48, k)
        s36_rest = engine.pack_sheets(unplaced, vw36, vh36) if unplaced else []
        placed_48 = sum(len(r["parts"]) for sh in s48_k for r in sh["rows"])
        placed_36 = sum(len(r["parts"]) for sh in s36_rest for r in sh["rows"])
        if placed_48 + placed_36 == n_req:
            out = []
            for i, sh in enumerate(s48_k):
                out.append({"id": len(out) + 1, "label": "4x8", "vw": vw48, "vh": vh48, "rows": sh["rows"]})
            for i, sh in enumerate(s36_rest):
                out.append({"id": len(out) + 1, "label": "3x6", "vw": vw36, "vh": vh36, "rows": sh["rows"]})
            candidates.append((out, total_area(out, lambda s: (s["vw"], s["vh"]))))

    # 混在: 3x6 を k 枚使ってから残りを 4x8
    for k in range(1, len(s36) + 1):
        s36_k, unplaced = engine.pack_sheets_max(parts_list, vw36, vh36, k)
        s48_rest = engine.pack_sheets(unplaced, vw48, vh48) if unplaced else []
        placed_36 = sum(len(r["parts"]) for sh in s36_k for r in sh["rows"])
        placed_48 = sum(len(r["parts"]) for sh in s48_rest for r in sh["rows"])
        if placed_36 + placed_48 == n_req:
            out = []
            for i, sh in enumerate(s36_k):
                out.append({"id": len(out) + 1, "label": "3x6", "vw": vw36, "vh": vh36, "rows": sh["rows"]})
            for i, sh in enumerate(s48_rest):
                out.append({"id": len(out) + 1, "label": "4x8", "vw": vw48, "vh": vh48, "rows": sh["rows"]})
            candidates.append((out, total_area(out, lambda s: (s["vw"], s["vh"]))))

    if not candidates:
        fallback = [{"id": i + 1, "label": "4x8", "vw": vw48, "vh": vh48, "rows": sh["rows"]} for i, sh in enumerate(s48)]
        return (fallback, total_area(fallback, lambda s: (s["vw"], s["vh"])))

    # 単一種の無駄面積（使用面積 − 部材総面積）。無駄が少ない板種を優先し、その板種だけで済むなら混在・他板種は選ばない。
    waste_36 = (area_36_only - parts_area) if area_36_only is not None else float("inf")
    waste_48 = (area_48_only - parts_area) if area_48_only is not None else float("inf")

    def score(item):
        sheets_list, area = item
        n36 = sum(1 for s in sheets_list if s.get("label") == "3x6")
        n48 = sum(1 for s in sheets_list if s.get("label") == "4x8")
        is_36_only = n36 > 0 and n48 == 0
        is_48_only = n48 > 0 and n36 == 0
        is_mixed = n36 > 0 and n48 > 0
        # 優先度: 無駄が少ない単一種 > 混在 > 無駄が多い単一種
        if is_36_only and waste_36 <= waste_48:
            tier = 0
        elif is_48_only and waste_48 <= waste_36:
            tier = 0
        elif is_mixed:
            tier = 1
        else:
            tier = 2
        # 同点時は小さい板(3x6)を優先
        prefer_small = 0 if is_36_only else (1 if is_48_only else 2)
        return (tier, area, len(sheets_list), prefer_small)

    best = min(candidates, key=score)
    return best


# --- FastAPI ---

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


class PartInput(BaseModel):
    """部品1つ分の入力"""
    n: str = Field(..., description="名称")
    w: float = Field(..., gt=0, description="幅 (mm)")
    d: float = Field(..., gt=0, description="奥行 (mm)")


class PackRequest(BaseModel):
    """木取り実行のリクエスト"""
    parts: list[PartInput] = Field(..., min_length=1, description="部品リスト")
    vw: float = Field(..., gt=0, description="定尺 長手 (mm)")
    vh: float = Field(..., gt=0, description="定尺 短手 (mm)")
    kerf: float = Field(3.0, ge=0, description="刃物厚 (mm)")
    label: str = Field("板", description="板のラベル（結果の表示用）")


class PackResponse(BaseModel):
    """木取り実行のレスポンス"""
    label: str
    vw: float
    vh: float
    sheet_count: int
    total_parts_placed: int
    total_parts_requested: int
    sheets: list


class PackAutoRequest(BaseModel):
    """自動選定（3x6・4x8・混在）のリクエスト"""
    parts: list[PartInput] = Field(..., min_length=1)
    vw36: float = Field(..., gt=0, description="3x6 長手 (mm)")
    vh36: float = Field(..., gt=0, description="3x6 短手 (mm)")
    vw48: float = Field(..., gt=0, description="4x8 長手 (mm)")
    vh48: float = Field(..., gt=0, description="4x8 短手 (mm)")
    kerf: float = Field(3.0, ge=0)


class SheetWithSize(BaseModel):
    """枚ごとにサイズを持つシート（混在用）"""
    id: int
    label: str
    vw: float
    vh: float
    rows: list


class PackAutoResponse(BaseModel):
    """自動選定のレスポンス（3x6と4x8の混在あり）"""
    sheet_count: int
    total_parts_placed: int
    total_parts_requested: int
    sheets: list  # 各要素は SheetWithSize 相当（label, vw, vh, rows）


class DiagramHtmlRequest(BaseModel):
    """木取図 印刷用 HTML 生成のリクエスト"""
    label: str
    vw: float
    vh: float
    sheets: list
    max_per_page: Optional[int] = Field(None, ge=1, description="1ページに載せる図の枚数。省略時は1枚ずつ")


app = FastAPI(
    title="イタドリ API",
    description="定尺板からの木取りネスティング・木取図生成 API",
    version="1.0.0",
)
# CORS: 全オリジン許可（preflight を通すため）。Cookie は使わないので credentials=False で可。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# preflight (OPTIONS) が 400 になる環境向け: OPTIONS を明示的に 200 で返す
@app.api_route("/api/{path:path}", methods=["OPTIONS"])
def options_api(path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.get("/")
def root():
    """API の説明とヘルス確認"""
    return {
        "service": "itadori",
        "message": "木取り API。POST /api/pack でネスティング、POST /api/diagram/html で印刷用HTML。",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/pack", response_model=PackResponse)
def api_pack(req: PackRequest):
    """部品リストを定尺板にネスティングし、シート割り当て結果を返す。"""
    engine = TrunkTechEngine(kerf=req.kerf)
    parts = [{"n": p.n, "w": p.w, "d": p.d} for p in req.parts]
    sheets = engine.pack_sheets(parts, req.vw, req.vh)
    total_placed = sum(len(r["parts"]) for s in sheets for r in s["rows"])
    return PackResponse(
        label=req.label,
        vw=req.vw,
        vh=req.vh,
        sheet_count=len(sheets),
        total_parts_placed=total_placed,
        total_parts_requested=len(parts),
        sheets=sheets,
    )


@app.post("/api/pack_auto", response_model=PackAutoResponse)
def api_pack_auto(req: PackAutoRequest):
    """3x6 のみ・4x8 のみ・3x6と4x8の混在 を試し、総使用面積が最小の案を返す。"""
    sheets, _area = find_best_mixed(
        req.parts, req.vw36, req.vh36, req.vw48, req.vh48, req.kerf
    )
    total_placed = sum(len(r["parts"]) for s in sheets for r in s["rows"])
    return PackAutoResponse(
        sheet_count=len(sheets),
        total_parts_placed=total_placed,
        total_parts_requested=len(req.parts),
        sheets=sheets,
    )


@app.post("/api/diagram/html", response_class=HTMLResponse)
def api_diagram_html(req: DiagramHtmlRequest):
    """木取図を印刷用 HTML で返す。"""
    best = {"label": req.label, "vw": req.vw, "vh": req.vh, "sheets": req.sheets}
    html = build_print_html(best, max_per_page=req.max_per_page)
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
