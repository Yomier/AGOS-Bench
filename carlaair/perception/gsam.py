"""Serve the frozen v13 visual grounding and six-view perception API."""
from __future__ import annotations
import argparse, base64, io, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _png_b64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GsamEngine:
    def __init__(self, detector_model, encoder_model, sam_model, device="auto",
                 box_threshold=0.20, text_threshold=0.18):
        import torch
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                                  AutoModel, Sam2Processor, Sam2Model)
        from PIL import Image
        self._torch = torch
        self._Image = Image
        self.device = ("cuda" if device == "auto" and torch.cuda.is_available()
                       else "cpu" if device == "auto" else device)
        self.detector_model = detector_model
        self.encoder_model = encoder_model
        self.sam_model_name = sam_model
        self.det_proc = AutoProcessor.from_pretrained(detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_model).to(self.device).eval()
        self.enc_proc = AutoProcessor.from_pretrained(encoder_model)
        self.enc = AutoModel.from_pretrained(encoder_model).to(self.device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(sam_model)
        self.sam = Sam2Model.from_pretrained(sam_model).to(self.device).eval()
        self._lock = threading.Lock()
        self._ref_cache = {}
        # 实际加载的 SAM 目录名（冻结记录用）
        try:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(sam_model, "config.json")
            self.sam_loaded_path = str(Path(p).parent) if p else sam_model
        except Exception:
            self.sam_loaded_path = sam_model

    # ---------- SigLIP2 ----------
    def _feat(self, img):
        t = self._torch
        inputs = self.enc_proc(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with t.no_grad():
            f = self.enc.get_image_features(**inputs)
        if not t.is_tensor(f):
            pooled = getattr(f, "pooler_output", None)
            f = pooled if pooled is not None else f.last_hidden_state[:, 0]
        return f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _ref_feat(self, ref_bytes: bytes):
        key = hash(ref_bytes)
        if key not in self._ref_cache:
            img = self._Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            self._ref_cache[key] = self._feat(img)
        return self._ref_cache[key]

    # ---------- 检测 + 参考匹配 + 可选分割 ----------
    def ground(self, *, image_bytes: bytes, ref_bytes: bytes, description: str,
               box_threshold: float, text_threshold: float, nms_iou: float,
               topk: int, segment: bool) -> dict:
        t0 = time.time()
        t = self._torch
        img = self._Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        with self._lock:
            query = (str(description or "vehicle").strip() + ". vehicle.")
            inp = self.det_proc(images=img, text=query, return_tensors="pt")
            inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
            with t.no_grad():
                out = self.det(**inp)
            try:
                proc = self.det_proc.post_process_grounded_object_detection(
                    out, inp.get("input_ids"), box_threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[[H, W]])[0]
            except TypeError:  # transformers>=5 renames box_threshold -> threshold
                proc = self.det_proc.post_process_grounded_object_detection(
                    out, inp.get("input_ids"), threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[[H, W]])[0]
            boxes = [[float(v) for v in b.tolist()] for b in proc.get("boxes", [])]
            scores = [float(s.item()) for s in proc.get("scores", [])]
            labels = [str(l) for l in proc.get("labels", [])]

            # NMS（IoU 阈值）后 topk，按 detector_score 降序
            keep = self._nms(boxes, scores, nms_iou)[: max(1, int(topk))]
            ref = self._ref_feat(ref_bytes)
            proposals = []
            for i in keep:
                bb = boxes[i]
                # DINO can predict beyond the raster (including a 1.34px
                # overflow in a real Dev frame). Crop and returned geometry
                # must use the same visible extent; do not publish an unclipped
                # box paired with an already clipped RGB crop.
                import math
                if len(bb) != 4 or not all(math.isfinite(v) for v in bb):
                    raise ValueError("invalid detector box geometry")
                bb = [max(0., min(v, float(W if j % 2 == 0 else H)))
                      for j, v in enumerate(bb)]
                if bb[2] <= bb[0] or bb[3] <= bb[1]:
                    continue
                crop = img.crop((max(0, int(bb[0])), max(0, int(bb[1])),
                                 min(W, int(bb[2]) + 1), min(H, int(bb[3]) + 1)))
                if crop.width < 2 or crop.height < 2:
                    continue
                cf = self._feat(crop)
                ident = max(0.0, min(1.0, (float((cf * ref).sum().item()) + 1.0) / 2.0))
                entry = {
                    "bbox_xyxy": bb, "detector_score": scores[i],
                    "identity_score": round(ident, 4),
                    "combined_score": round(0.35 * scores[i] + 0.65 * ident, 4),
                    "label": labels[i] if i < len(labels) else "",
                    "crop_b64": _png_b64(crop),
                }
                if segment:
                    m = self._segment(img, bb)
                    if m is not None:
                        mask_arr, mb, area = m
                        entry["mask_bbox_xyxy"] = mb
                        entry["mask_area_px"] = int(area)
                        # 去背景分割图：全图 mask 裁到检测框区域后应用于 crop
                        import numpy as _np
                        bx1, by1 = max(0, int(bb[0])), max(0, int(bb[1]))
                        mask_crop = mask_arr[by1:by1 + crop.height, bx1:bx1 + crop.width]
                        if mask_crop.shape[:2] != (crop.height, crop.width):
                            mask_img = self._Image.fromarray(mask_crop).resize(
                                (crop.width, crop.height))
                        else:
                            mask_img = self._Image.fromarray(mask_crop)
                        canvas = self._Image.new("RGB", crop.size, (0, 0, 0))
                        canvas.paste(crop, (0, 0), mask_img)
                        entry["masked_crop_b64"] = _png_b64(canvas)
                    else:
                        entry["mask_bbox_xyxy"] = None  # 分割失败：客户端剔除该 proposal
                proposals.append(entry)
            # combined 降序输出（客户端稳定编号 P1..）
            proposals.sort(key=lambda z: z.get("combined_score", 0.0), reverse=True)
        return {"status": "ok" if proposals else "empty",
                "sam_model": self.sam_model_name,
                "sam_loaded_path": self.sam_loaded_path,
                "detector_model": self.detector_model,
                "encoder_model": self.encoder_model,
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
                "proposals": proposals}

    def _nms(self, boxes, scores, iou_thr: float):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        for i in order:
            ok = True
            for j in keep:
                if self._iou(boxes[i], boxes[j]) > iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(i)
        return keep

    # ------------------------------------------------------------------
    # UGV 六视图证据板（2026-10-11 设计 v2 §4.2；POST /v1/ugv_views）。
    # 顺序固定：合法框检查 → 同panel NMS → 每panel检测分Top(panel_limit)
    # → 按V编号及框坐标分配稳定ID → 完整单参考 SigLIP2 排序（仅原始cosine，
    # 不混合检测分）→ panel多样性Top(display_limit)。检测 query 是通用车辆
    # 词表，不把目标颜色作为检测条件。segment 固定忽略（设计裁定 SAM 关闭）。
    # ------------------------------------------------------------------
    def ground_ugv_six(self, *, bundle_id: str, views: list, ref_bytes: bytes,
                       detection_query: str, box_threshold: float,
                       text_threshold: float, nms_iou: float,
                       panel_limit: int, display_limit: int,
                       evidence_id_mode: str = "full") -> dict:
        import math as _math
        if evidence_id_mode not in ("full", "local_card_v1"):
            raise ValueError("unsupported evidence_id_mode")
        t0 = time.time()
        import hashlib as _hashlib
        Image = self._Image
        panels_out = []
        proposals = []
        ok_panels = 0
        with self._lock:
            ref = self._ref_feat(ref_bytes)
            query = str(detection_query or "car . van . truck . bus . vehicle .").strip()
            if not query.endswith("."):
                query += "."
            for v in (views or [])[:6]:
                vid = str(v.get("view_id") or "")
                pstat = {"view_id": vid, "status": "ok", "n_raw": 0,
                         "source_sha256": "",
                         "n_invalid_box": 0, "n_after_nms": 0, "n_after_topk": 0,
                         "invalid_reasons": []}
                try:
                    raw = base64.b64decode(v.get("image_b64") or "", validate=True)
                except Exception:
                    pstat.update({"status": "invalid_source", "invalid_reasons": ["bad_b64"]})
                    panels_out.append(pstat)
                    continue
                # Hash decoded bytes even on rejected panels. Never echo the
                # requested hash as evidence of what the server actually read.
                pstat["source_sha256"] = _hashlib.sha256(raw).hexdigest()
                if not v.get("valid", True) or not raw:
                    pstat.update({"status": "invalid_source", "invalid_reasons": ["view_invalid"]})
                    panels_out.append(pstat)
                    continue
                src_sha = str(v.get("source_sha256") or "")
                if src_sha and _hashlib.sha256(raw).hexdigest() != src_sha:
                    pstat.update({"status": "invalid_source",
                                  "invalid_reasons": ["source_hash_mismatch"]})
                    panels_out.append(pstat)
                    continue
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    pstat.update({"status": "invalid_source", "invalid_reasons": ["bad_image"]})
                    panels_out.append(pstat)
                    continue
                W, H = img.size
                inp = self.det_proc(images=img, text=query, return_tensors="pt")
                inp = {k: (val.to(self.device) if hasattr(val, "to") else val) for k, val in inp.items()}
                with self._torch.no_grad():
                    out = self.det(**inp)
                try:
                    proc = self.det_proc.post_process_grounded_object_detection(
                        out, inp.get("input_ids"), box_threshold=box_threshold,
                        text_threshold=text_threshold, target_sizes=[[H, W]])[0]
                except TypeError:
                    proc = self.det_proc.post_process_grounded_object_detection(
                        out, inp.get("input_ids"), threshold=box_threshold,
                        text_threshold=text_threshold, target_sizes=[[H, W]])[0]
                boxes = [[float(bv) for bv in b.tolist()] for b in proc.get("boxes", [])]
                scores = [float(s.item()) for s in proc.get("scores", [])]
                pstat["n_raw"] = len(boxes)
                # 1) 合法框检查：有限数值 + clip 后 floor/ceil 宽高≥2px
                legal = []
                for bb, sc in zip(boxes, scores):
                    if len(bb) != 4 or not all(_math.isfinite(x) for x in bb):
                        pstat["n_invalid_box"] += 1
                        pstat["invalid_reasons"].append("INVALID_BOX")
                        continue
                    cx1 = max(0, int(_math.floor(max(0.0, bb[0]))))
                    cy1 = max(0, int(_math.floor(max(0.0, bb[1]))))
                    cx2 = min(W, int(_math.ceil(min(float(W), bb[2]))))
                    cy2 = min(H, int(_math.ceil(min(float(H), bb[3]))))
                    if cx2 - cx1 < 2 or cy2 - cy1 < 2 or bb[2] <= bb[0] or bb[3] <= bb[1]:
                        pstat["n_invalid_box"] += 1
                        pstat["invalid_reasons"].append("INVALID_BOX")
                        continue
                    legal.append(([cx1, cy1, cx2, cy2], sc))
                # 2) 同 panel NMS（不同 panel 像素坐标系不同，不做跨图 NMS）
                if legal:
                    lb = [l[0] for l in legal]
                    ls = [l[1] for l in legal]
                    keep = self._nms(lb, ls, nms_iou)
                    legal = [legal[i] for i in keep]
                pstat["n_after_nms"] = len(legal)
                # 3) 每 panel 检测分 Top(panel_limit)
                legal.sort(key=lambda l: -l[1])
                legal = legal[: max(1, int(panel_limit))]
                pstat["n_after_topk"] = len(legal)
                # 4) 稳定 ID：NMS 后、参考排序前，按框坐标字典序分配 Pm
                legal.sort(key=lambda l: (l[0][0], l[0][1], l[0][2], l[0][3]))
                if legal:
                    ok_panels += 1
                panels_out.append(pstat)  # 逐 panel 状态记录（成功路径同样必须落）
                for m, (bb, sc) in enumerate(legal, start=1):
                    crop = img.crop((bb[0], bb[1], bb[2], bb[3]))  # raw crop：无 padding
                    cf = self._feat(crop)
                    cos = float((cf * ref).sum().item())  # 原始 cosine 完整精度
                    proposals.append({
                        "proposal_id": "%s/%s/P%d" % (bundle_id, vid, m),
                        "view_id": vid,
                        "bbox_xyxy": [float(x) for x in bb],
                        "detector_score": float(sc),
                        "ref_cosine": cos,
                        "raw_crop_b64": _png_b64(crop),
                        "raw_crop_sha256": _hashlib.sha256(crop.tobytes()).hexdigest(),
                        "_source_b64": base64.b64encode(raw).decode("ascii"),
                        "view_width": W, "view_height": H,
                        "panel_rank": None,
                    })
            # 5) 全局排序：仅原始 cosine 降序；并列按检测分降序 → V 编号 → 框坐标
            vidx = {v: i for i, v in enumerate(
                [p.get("view_id") for p in (views or [])[:6]])}
            proposals.sort(key=lambda p: (
                -p["ref_cosine"], -p["detector_score"],
                vidx.get(p["view_id"], 99),
                p["bbox_xyxy"][0], p["bbox_xyxy"][1],
                p["bbox_xyxy"][2], p["bbox_xyxy"][3]))
            # 6) panel 多样性展示选择：每 panel 最佳代表按全局排序取至多
            #    display_limit；不足再从剩余框按同一排序补齐。
            shown, panel_best_seen = [], {}
            for i, p in enumerate(proposals):
                p["global_rank"] = i + 1
            for p in proposals:
                if p["view_id"] not in panel_best_seen:
                    panel_best_seen[p["view_id"]] = p
            for p in sorted(panel_best_seen.values(),
                            key=lambda z: z["global_rank"])[: max(1, int(display_limit))]:
                shown.append(p)
            if len(shown) < max(1, int(display_limit)):
                chosen = set(id(p) for p in shown)
                for p in proposals:
                    if len(shown) >= max(1, int(display_limit)):
                        break
                    if id(p) not in chosen:
                        shown.append(p)
            shown_ids = set(id(p) for p in shown)
            for p in proposals:
                p["shown"] = id(p) in shown_ids
            if evidence_id_mode == "local_card_v1":
                # Labels identify cards only within this response. Full IDs and
                # source hashes stay in metadata; the VLM need not reproduce them.
                for i, p in enumerate(shown, start=1):
                    p["model_id"] = "C%d" % i
            # 7) 证据卡与板渲染（20% padding、长边256、pad 256×256、顶栏32px）
            board_b64 = ""
            cards = []
            if shown:
                cards = [self._render_evidence_card(p) for p in shown]
                board_b64 = self._render_board(cards)
            # 状态机：invalid_source panel 存在 → partial_error（有 proposal）/
            # error（无）；否则 empty / ok。检测器异常由上层 except 兜底为 error。
            has_invalid = any(p["status"] == "invalid_source" for p in panels_out)
            if has_invalid:
                status = "partial_error" if proposals else "error"
            else:
                status = "ok" if proposals else "empty"
        result = {
            "status": status,
            "bundle_id": bundle_id,
            "evidence_id_mode": evidence_id_mode,
            "panels": panels_out,
            # 全量 proposal 记录（原始 cosine 完整精度）；raw_crop_b64 仅 shown
            # 保留（审计/复算），其余仅留 hash（响应体积合同）。
            "proposals": [
                {**{k: v for k, v in p.items() if k not in ("_source_b64", "raw_crop_b64")},
                 **({"raw_crop_b64": p["raw_crop_b64"]} if p.get("shown") else {})}
                for p in proposals],
            "cards": cards,
            "board_b64": board_b64,
            "board_sha256": (_hashlib.sha256(board_b64.encode("ascii")).hexdigest()
                             if board_b64 else ""),
            "n_shown": len(shown),
            "detector_model": self.detector_model,
            "encoder_model": self.encoder_model,
            "sam_used": False,
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
        }
        return result

    def _render_evidence_card(self, prop: dict):
        """单张证据卡：20% 上下文 padding 的裁剪 → 长边256 → pad 256×256 →
        顶栏 32px（proposal ID + view 名）。标签在车辆图外。"""
        Image = self._Image
        try:
            crop = Image.open(io.BytesIO(base64.b64decode(prop["raw_crop_b64"]))).convert("RGB")
        except Exception:
            crop = Image.new("RGB", (64, 64), (32, 32, 32))
        W, H = prop["view_width"], prop["view_height"]
        x1, y1, x2, y2 = prop["bbox_xyxy"]
        bw, bh = x2 - x1, y2 - y1
        px1 = max(0, int(x1 - 0.2 * bw)); py1 = max(0, int(y1 - 0.2 * bh))
        px2 = min(W, int(x2 + 0.2 * bw) + 1); py2 = min(H, int(y2 + 0.2 * bh) + 1)
        # raw_crop_b64 是无 padding 的检测框 crop；证据卡需要源 panel 上的
        # padding 版本——此处从 crop 反推等比例外扩（服务端无原图引用时退化为
        # 直接用 crop 本身，板仍保留原始框坐标说明）。
        try:
            full = Image.open(io.BytesIO(base64.b64decode(prop.get("_source_b64", "")))).convert("RGB")
            padded = full.crop((px1, py1, px2, py2))
        except Exception:
            padded = crop
        scale = 256.0 / max(padded.width, padded.height, 1)
        nw, nh = max(1, int(round(padded.width * scale))), max(1, int(round(padded.height * scale)))
        resized = padded.resize((nw, nh))
        canvas = Image.new("RGB", (256, 256), (16, 16, 16))
        canvas.paste(resized, ((256 - nw) // 2, (256 - nh) // 2))
        from PIL import ImageDraw, ImageFont
        card = Image.new("RGB", (256, 288), (24, 24, 24))
        card.paste(canvas, (0, 32))
        d = ImageDraw.Draw(card)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # 完整 proposal ID（B…/Vn/Pm）：图上标签必须与 JSON 合同同串，
        # 模型按图复述的 ID 才能通过触发/核验合法性校验
        label = "%s [%s]" % (prop.get("model_id") or prop["proposal_id"], prop["view_id"])
        if prop.get("model_id"):
            try:
                font = ImageFont.load_default(size=20)
            except TypeError:
                pass
        d.text((4, 4), label[:46], fill=(255, 220, 80), font=font)
        buf = io.BytesIO()
        card.save(buf, format="PNG")
        return {"proposal_id": prop["proposal_id"], "view_id": prop["view_id"],
                "model_id": prop.get("model_id"),
                "card_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "native_box_size": [int(x2 - x1), int(y2 - y1)],
                "crop_range": [int(px1), int(py1), int(px2), int(py2)],
                "scale_ratio": round(float(scale), 4)}

    def _render_board(self, cards: list):
        """一行 ≤display_limit 卡；总尺寸至多 768×288；不足不填假 proposal。"""
        Image = self._Image
        n = len(cards)
        if n == 0:
            return ""
        board = Image.new("RGB", (256 * n, 288), (8, 8, 8))
        for i, c in enumerate(cards):
            try:
                card = Image.open(io.BytesIO(base64.b64decode(c["card_b64"]))).convert("RGB")
                board.paste(card, (256 * i, 0))
            except Exception:
                continue
        buf = io.BytesIO()
        board.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + \
             max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _segment(self, img, bbox):
        """SAM2 box-prompt 分割。返回 (mask_np uint8 0/255, mask_bbox, area) 或 None。
        transformers 5.x API：post_process_masks(pred_masks, original_sizes) →
        list[tensor(num_masks, 3层级, H, W)]；用 iou_scores 选最优层级。"""
        import numpy as np
        t = self._torch
        try:
            box = [[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]]
            inputs = self.sam_proc(images=img, input_boxes=[box], return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with t.no_grad():
                out = self.sam(**inputs)
            masks = self.sam_proc.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu())
            m = masks[0]                                   # [num_masks(=1), 3, H, W]
            ious = out.iou_scores.cpu()                   # [batch(=1), num_masks, 3]
            lvl = int(ious[0, 0].argmax().item())
            best = m[0, lvl]                              # [H, W]
            arr = (best.numpy() if hasattr(best, "numpy") else np.asarray(best)).astype("uint8") * 255
            ys, xs = (arr > 0).nonzero()
            if len(xs) == 0:
                return None
            return arr, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(len(xs))
        except Exception as e:
            import traceback
            sys.stderr.write("[_segment] %s: %s\n%s\n" % (
                type(e).__name__, e, traceback.format_exc()))
            sys.stderr.flush()
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8201)
    ap.add_argument("--detector-model", required=True)
    ap.add_argument("--encoder-model", required=True)
    ap.add_argument("--sam-model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--preload-warmup-image", default="",
                    help="可选：预热用图片路径（正式跑前预热 CUDA kernel）")
    a = ap.parse_args()

    for path in (a.detector_model, a.encoder_model, a.sam_model):
        if not Path(path).is_dir():
            ap.error(f"model directory does not exist: {path}")
    print(f"Loading models on device=auto ...", flush=True)
    eng = GsamEngine(a.detector_model, a.encoder_model, a.sam_model, a.device)
    print(f"Loaded: det={a.detector_model} enc={a.encoder_model} sam={eng.sam_loaded_path} "
          f"device={eng.device}", flush=True)
    if a.preload_warmup_image and Path(a.preload_warmup_image).exists():
        from PIL import Image
        import numpy as np
        buf = io.BytesIO()
        Image.fromarray(np.zeros((576, 1024, 3), dtype="uint8")).save(buf, format="PNG")
        eng.ground(image_bytes=buf.getvalue(), ref_bytes=buf.getvalue(),
                   description="vehicle", box_threshold=0.2, text_threshold=0.18,
                   nms_iou=0.5, topk=1, segment=True)
        print("Warmup OK", flush=True)

    class H(BaseHTTPRequestHandler):
        server_version = "GSAM-Sidecar/1"
        def log_message(self, fmt, *args):
            sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))
        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._j(200, {"ok": True, "service": "gsam-sidecar",
                                     "sam_model": eng.sam_model_name,
                                     "sam_loaded_path": eng.sam_loaded_path,
                                     "device": eng.device})
            return self._j(404, {"ok": False, "error": "not_found"})
        def do_POST(self):
            if self.path == "/v1/ugv_views":
                try:
                    n = int(self.headers.get("Content-Length", "0") or 0)
                    if n <= 0 or n > 120 * 1024 * 1024:
                        raise ValueError("invalid Content-Length")
                    q = json.loads(self.rfile.read(n).decode("utf-8"))
                    if str(q.get("domain") or "") != "ugv":
                        return self._j(400, {"status": "error",
                                             "error": "domain must be 'ugv' for this endpoint"})
                    views = q.get("views") or []
                    if not isinstance(views, list) or not views or len(views) > 6:
                        return self._j(400, {"status": "error",
                                             "error": "views must be a list of 1..6 entries"})
                    ref = base64.b64decode(q.get("target_reference_b64") or "", validate=True)
                    if not ref:
                        return self._j(400, {"status": "error",
                                             "error": "target_reference_b64 is required"})
                    r = eng.ground_ugv_six(
                        bundle_id=str(q.get("bundle_id") or "bundle"),
                        views=views, ref_bytes=ref,
                        detection_query=str(q.get("detection_query") or ""),
                        box_threshold=float(q.get("box_threshold") or 0.20),
                        text_threshold=float(q.get("text_threshold") or 0.18),
                        nms_iou=float(q.get("nms_iou") or 0.50),
                        panel_limit=int(q.get("panel_limit") or 8),
                        display_limit=int(q.get("display_limit") or 3),
                        evidence_id_mode=str(q.get("evidence_id_mode") or "full"))
                    return self._j(200, r)
                except Exception as e:
                    import traceback as _tb
                    sys.stderr.write("[ugv_views] %s: %s\n%s\n" % (
                        type(e).__name__, e, _tb.format_exc()))
                    sys.stderr.flush()
                    return self._j(400, {"status": "error",
                                         "error": f"{type(e).__name__}: {e}"})
            if self.path != "/v1/gsam":
                return self._j(404, {"status": "error", "error": "not_found"})
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n <= 0 or n > 60 * 1024 * 1024:
                    raise ValueError("invalid Content-Length")
                q = json.loads(self.rfile.read(n).decode("utf-8"))
                image = base64.b64decode(q.get("image_b64") or "", validate=True)
                ref = base64.b64decode(q.get("target_reference_b64") or "", validate=True)
                if not image or not ref:
                    raise ValueError("image_b64 and target_reference_b64 are required")
                r = eng.ground(
                    image_bytes=image, ref_bytes=ref,
                    description=str(q.get("target_description") or "vehicle"),
                    box_threshold=float(q.get("box_threshold") or 0.20),
                    text_threshold=float(q.get("text_threshold") or 0.18),
                    nms_iou=float(q.get("nms_iou") or 0.50),
                    topk=int(q.get("topk") or 3),
                    segment=bool(q.get("segment")))
                return self._j(200, r)
            except Exception as e:
                return self._j(400, {"status": "error", "error": f"{type(e).__name__}: {e}"})
        def _j(self, status, obj):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"GSAM sidecar: http://{a.host}:{a.port}/v1/gsam", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
