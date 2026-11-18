"""Action validation and the OpenAI-compatible VLM transport."""
import json, re, logging, requests, time, math
from .http_accounting import record_http_attempt
from typing import Optional, Tuple

logger = logging.getLogger("LiteV1.Agents")

def repair_json(raw_text):
    """尝试从 VLM 输出中提取 JSON"""
    text = raw_text.strip()
    # Remove markdown
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    # Fast path: a bare JSON object may nest deeper than the bounded search below.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except:
        pass
    # Try adding closing braces
    for _ in range(3):
        text += "}"
        try:
            return json.loads(text)
        except:
            pass
    return None

# ============================================================
# Validator
# ============================================================
GRID_CELLS = {f"{r}{c}" for r in "ABCDE" for c in range(1, 6)}

def model_field_error(parsed):
    """Validate model-owned values before action construction or consumption.

    This checks representation only; menus, candidate state and visual protocol
    semantics remain with their existing validators.
    """
    if not isinstance(parsed, dict):
        return "invalid_field: response must be an object"
    def finite_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    optional_strings = {
        'waypoint_id', 'fallback_waypoint_id', 'candidate_id', 'grid_cell',
        'geometry_type', 'source_image_sha256', 'localization_source',
        'target_verdict', 'selected_proposal', 'evidence_view', 'proposal_id',
        'source_proposal_id', 'fallback_action',
    }
    for name in ('role', 'action', 'message', 'reason', 'visual_description',
                 'evidence', *sorted(optional_strings)):
        if name not in parsed or (name in optional_strings and parsed[name] is None):
            continue
        # Cooperative methods group model fields under {"decision":..., "action":...};
        # nested dicts are validated by that method's own schema, not by the flat
        # legacy representation checks below.
        if isinstance(parsed.get(name), dict) and isinstance(parsed.get('decision'), dict):
            continue
        if not isinstance(parsed[name], str):
            return f"invalid_field: {name} must be a string"
    if 'confidence' in parsed:
        value = parsed['confidence']
        if not finite_number(value):
            return "invalid_field: confidence must be a finite number"
    if parsed.get('position_xy') is not None:
        xy = parsed['position_xy']
        if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                or any(not finite_number(v) for v in xy)):
            return "invalid_field: position_xy must contain two finite numbers"
    return ""


def validate_action(parsed, obs, available_waypoint_ids):
    """校验 VLM 输出的动作。首先检查 action 是否在 obs.available_actions 中。"""
    if not parsed:
        return False, "json_parse_failed"
    field_error = model_field_error(parsed)
    if field_error:
        return False, field_error

    role = parsed.get("role", "")
    action = parsed.get("action", "")

    if role != obs.role:
        return False, f"wrong_role: expected {obs.role}, got {role}"

    if action not in obs.available_actions:
        # GSAM T1（2026-10-10 设计 §4）：SUSPECT_CANDIDATE 菜单模式下，最终
        # ANNOTATE_CANDIDATE 只能由 method 层 gsam pipeline 转换产出（point 合同
        # 或网格回退 region）——模型自身无法直接输出 ANNOTATE（不在菜单），此处
        # 豁免菜单检查不引入越权通道。
        _gsam_final_annotate = (
            action == "ANNOTATE_CANDIDATE"
            and "SUSPECT_CANDIDATE" in (obs.available_actions or [])
            and str(parsed.get("geometry_type") or "") == "point")
        if not _gsam_final_annotate:
            return False, f"action_not_available: {action} not in {obs.available_actions}"

    if obs.role == "UAV":
        if action == "MOVE_TO_UAV_WAYPOINT":
            wp_id = parsed.get("waypoint_id", "")
            if wp_id not in available_waypoint_ids:
                return False, f"invalid_waypoint: {wp_id} not in {available_waypoint_ids}"
            from agos.benchmark_region import observation_region, inside_task_region
            bounds = observation_region(obs)
            if bounds is not None:
                wp = next((w for w in obs.available_waypoints if w.get('waypoint_id') == wp_id), None)
                if wp is None or not inside_task_region(wp.get('world_xy', ()), bounds):
                    return False, 'invalid_waypoint_region: outside task region or missing coordinates'
        if action == "ANNOTATE_CANDIDATE":
            # No annotation allowed when active candidate exists (also enforced in CandidateLedger)
            if obs.active_candidate is not None:
                if bool(getattr(obs, "pace_q_enabled", False)):
                    # R16-Q: annotation allowed while UGV verifies, as long as the
                    # shadow queue has room. Ledger re-checks on execute.
                    n_shadow = int(getattr(obs, "pace_shadow_count", 0) or 0)
                    max_shadow = int(getattr(obs, "pace_q_max_shadow", 2) or 2)
                    if n_shadow >= max_shadow:
                        return False, "shadow_queue_full"
                else:
                    return False, "active_candidate_pending_ugv_verification"
            # GSAM 两实验（2026-10-10 设计 §9.1）：点类型 ANNOTATE 合同——
            # geometry_type='point' 时 grid_cell 可空，position_xy 必填。
            from agos.methods.common.uav_annotation import finite_xy
            if str(parsed.get("geometry_type") or "") != "point":
                return False, "point_candidate_required: grid annotation retired"
            if not finite_xy(parsed.get("position_xy")):
                return False, "invalid_point_position"
            if parsed.get("grid_cell"):
                return False, "grid_annotation_retired"
            if obs.annotation_count >= obs.max_annotations:
                return False, "annotation_limit_reached"

    elif obs.role == "UGV":
        if action == "MOVE_TO_ROAD_WAYPOINT":
            wp_id = parsed.get("waypoint_id", "")
            if wp_id not in available_waypoint_ids:
                return False, f"invalid_waypoint: {wp_id} not in {available_waypoint_ids}"
        if action == "GO_TO_CANDIDATE":
            cid = parsed.get("candidate_id", "")
            if not cid:
                return False, "missing_candidate_id for GO_TO_CANDIDATE"
            if obs.active_candidate is None:
                return False, "no_active_candidate"
            if obs.active_candidate.candidate_id != cid:
                return False, f"candidate_id mismatch: {cid} vs active {obs.active_candidate.candidate_id}"
            if obs.active_candidate.status != "unverified":
                return False, f"candidate not unverified: status={obs.active_candidate.status}"
        if action in ("CONFIRM", "REJECT"):
            # 2026-09-04（用户裁定）：CONFIRM/REJECT 任何时候都允许——六视图全程常态化，
            # 不再要求 fresh STOP context。仅保留预算与 candidate_id 一致性校验。
            if action == "CONFIRM":
                if obs.confirm_budget <= 0:
                    return False, "confirm_budget_exhausted"
            # R2-P2-04: CONFIRM 与 REJECT 同等校验 candidate_id 一致性
            # （省略 id 允许：runner 会 auto-bind active candidate 或按 independent 处理）
            cid = parsed.get("candidate_id")
            if cid and obs.active_candidate and obs.active_candidate.candidate_id != cid:
                return False, f"{action.lower()}_candidate_id mismatch: {cid} vs active {obs.active_candidate.candidate_id}"
            # REJECT does not consume confirm budget

    return True, "valid"

# ============================================================
# VLM Client
# ============================================================
# R2-P1-07（2026-09-04 外审）：SIGTERM 快速取消机制。
# 旧实现 retry backoff 整段 time.sleep(30s)，supervisor 发 SIGTERM 后
# 子进程最长还要等一个 backoff 周期才回到 runner 循环顶部，15s grace 内
# 写不出 partial result。现引入模块级 cancel flag：
# - runner 的 SIGTERM handler 调 request_cancel()；
# - retry sleep 分片为 0.5s 步进并检查 flag，命中即抛 VLMCancelledError；
# - runner 捕获后走 EPISODE_WALL_TIMEOUT 收尾路径写 result。
_CANCEL_FLAG = {"flag": False}


class VLMCancelledError(RuntimeError):
    """SIGTERM 取消：VLM 调用被墙钟超时中断（非模型/接口错误）。"""


def request_cancel():
    _CANCEL_FLAG["flag"] = True


def clear_cancel():
    _CANCEL_FLAG["flag"] = False


def _sleep_cancellable(wait_s: float):
    """分片 sleep：每 0.5s 检查一次 cancel flag，命中抛 VLMCancelledError。"""
    import time as _time
    end = _time.time() + wait_s
    while True:
        if _CANCEL_FLAG["flag"]:
            raise VLMCancelledError("VLM call cancelled by SIGTERM (wall-clock timeout)")
        remain = end - _time.time()
        if remain <= 0:
            return
        _time.sleep(min(0.5, remain))


class VLMClient:
    def __init__(self, config, timeout_s=None, max_retries=None, top_p=None, seed=None):
        self.api_url = config.get("api_url", "")
        import os as _os
        self.api_key = _os.environ.get(config.get("api_key_env", "DASHSCOPE_API_KEY"), config.get("api_key", ""))
        self.model = config.get("model", "qwen3-vl-plus")
        self.timeout = timeout_s if timeout_s is not None else config.get("timeout_s", 120)
        self.temperature = config.get("temperature", 0.0)
        self._require_zero_temperature()
        self.max_tokens = config.get("max_output_tokens", 800)
        self.max_retries = max_retries if max_retries is not None else config.get("max_retries_per_call", config.get("max_retries", 3))
        self.retry_backoff_base = config.get("retry_backoff_base", 2.0)
        self.retry_backoff_max = config.get("retry_backoff_max", 30.0)
        # 2026-10-18: finite transport recovery within one logical call. Legacy
        # wait_forever cannot override either the count or elapsed-time limit.
        self.wait_forever = False
        self.total_timeout_s = config.get("total_timeout_s", 180.0)
        import math
        for name, value in (("timeout_s", self.timeout), ("total_timeout_s", self.total_timeout_s)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.top_p = top_p if top_p is not None else config.get("top_p", None)
        self.seed = seed if seed is not None else config.get("seed", None)
        self.reasoning_effort = config.get("reasoning_effort", None)
        self.chat_template_kwargs = config.get("chat_template_kwargs", None)
        self.response_format = config.get("response_format", None)
        # 只读元数据：最近一次成功 chat 的 finish_reason（""=未知/未记录）
        self.last_finish_reason = ""
        self.last_http_attempts = 0
        self.last_prompt_tokens = None
        self.last_chat_payload_sha256 = None
        self.last_chat_config_sha256 = None

    def _require_zero_temperature(self):
        if type(self.temperature) not in (int, float) or self.temperature != 0:
            raise ValueError("experimental base model temperature must be 0")

    def _retry_wait(self, attempt, started, base=None):
        """Bound retry scheduling; requests' timeout remains a connect/read limit.

        The episode process supervisor supplies the outer wall-clock deadline.
        No new POST is started after cancellation or the logical time budget.
        """
        import time
        if _CANCEL_FLAG['flag']:
            raise VLMCancelledError('VLM retry cancelled')
        remaining = self.total_timeout_s - (time.monotonic() - started)
        if attempt >= self.max_retries or remaining <= 0:
            return False
        base = self.retry_backoff_base if base is None else base
        wait = min(base ** (attempt + 1), self.retry_backoff_max)
        if wait >= remaining:
            return False
        logger.warning('VLM transport retry %s/%s after %.1fs', attempt + 1, self.max_retries, wait)
        _sleep_cancellable(wait)
        return True

    @staticmethod
    def _payload_sha256(value):
        import hashlib
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                                         ensure_ascii=True).encode('utf-8')).hexdigest()

    def _chat_config_sha256(self):
        # Bind endpoint, decoding, template and authorization without exposing
        # the secret in the receipt. Used only by the exact PV preflight.
        return self._payload_sha256({key:getattr(self,key,None) for key in
            ('api_url','api_key','model','timeout','total_timeout_s','max_retries','temperature','max_tokens','top_p',
             'seed','reasoning_effort','chat_template_kwargs','response_format')})

    def prepare_chat_payload(self, system_prompt, user_prompt, images_base64=None):
        """Single message builder shared by generation and served tokenization."""
        self._require_zero_temperature()
        import copy
        content = []
        if images_base64:
            if isinstance(images_base64[0], tuple):
                for label,img_b64 in images_base64:
                    content.append({'type':'text','text':str(label)})
                    content.append({'type':'image_url','image_url':{'url':f'data:image/png;base64,{img_b64}'}})
            else:
                for img_b64 in images_base64:
                    content.append({'type':'image_url','image_url':{'url':f'data:image/png;base64,{img_b64}'}})
        content.append({'type':'text','text':user_prompt})
        payload={'model':self.model,'messages':[{'role':'system','content':system_prompt},
            {'role':'user','content':content}],'temperature':self.temperature,'max_tokens':self.max_tokens}
        template=getattr(self,'chat_template_kwargs',None)
        if isinstance(template,dict) and template:
            payload['chat_template_kwargs']=copy.deepcopy(template)
        if self.response_format is not None:
            payload['response_format']=copy.deepcopy(self.response_format)
        for key in ('top_p','seed','reasoning_effort'):
            value=getattr(self,key,None)
            if value is not None:payload[key]=value
        return payload

    def preflight_chat_tokens(self, system_prompt, user_prompt, images_base64=None):
        """Exact /tokenize count for identical full messages; no generation.

        One request, no retry or alternate tokenizer. Formal PV requires this
        for every nonempty batch; other PV configurations use it on estimate overflow.
        This cannot consume a model
        generation call and does not change generation HTTP accounting.
        """
        from urllib.parse import urlsplit,urlunsplit
        import time
        if _CANCEL_FLAG['flag']:
            raise VLMCancelledError('VLM tokenization cancelled')
        url=urlsplit(self.api_url)
        if (url.scheme not in ('http','https') or not url.netloc or
                not url.path.endswith('/v1/chat/completions') or url.query or url.fragment):
            raise RuntimeError('tokenizer endpoint binding requires same served chat endpoint')
        endpoint=urlunsplit((url.scheme,url.netloc,url.path[:-len('/v1/chat/completions')]+'/tokenize','',''))
        config_sha=self._chat_config_sha256()
        generation=self.prepare_chat_payload(system_prompt,user_prompt,images_base64)
        token_payload={k:generation[k] for k in ('model','messages','chat_template_kwargs') if k in generation}
        token_payload['add_generation_prompt']=True
        started=time.perf_counter()
        response=requests.post(endpoint,json=token_payload,headers={
            'Content-Type':'application/json','Authorization':f'Bearer {self.api_key}'},
            timeout=self.timeout,allow_redirects=False)
        if response.status_code!=200:
            raise RuntimeError('tokenizer HTTP failure: '+str(response.status_code))
        data=response.json()
        if not isinstance(data,dict):raise RuntimeError('tokenizer response is not an object')
        count,limit,tokens=data.get('count'),data.get('max_model_len'),data.get('tokens')
        if (type(count) is not int or count<=0 or type(limit) is not int or limit<=0 or
                not isinstance(tokens,list) or len(tokens)!=count or
                not all(type(t) is int and t>=0 for t in tokens)):
            raise RuntimeError('tokenizer count/token sequence mismatch')
        if data.get('model',self.model)!=self.model:
            raise RuntimeError('tokenizer model binding mismatch')
        if (config_sha!=self._chat_config_sha256() or
                generation!=self.prepare_chat_payload(system_prompt,user_prompt,images_base64)):
            raise RuntimeError('tokenizer request/config binding changed')
        receipt={'count':count,'max_model_len':limit,'model':self.model,
            'chat_endpoint':self.api_url,'tokenize_endpoint':endpoint,
            'payload_sha256':self._payload_sha256(generation),
            'messages_sha256':self._payload_sha256(generation['messages']),
            'tokenize_payload_sha256':self._payload_sha256(token_payload),
            'config_sha256':config_sha,'generation_calls':0,
            'latency_ms':(time.perf_counter()-started)*1000}
        receipt['receipt_sha256']=self._payload_sha256(receipt)
        return receipt

    def chat(self, system_prompt, user_prompt, images_base64=None):
        """调用 VLM API，返回结构化结果，区分成功与各类错误。

        鲁棒重试策略（2026-07-26）：
        - 429/5xx/超时/网络错误：指数退避，受次数和累计180s调度预算双限
        - 503 特殊处理：中转站临时过载，backoff 更长（×1.5）
        - 400 参数错误：不重试（重试无用），记录详细 body 供排查
        - 401/403 认证错误：立即 fail-fast
        - 单次退避上限 retry_backoff_max（默认30s），避免过长等待
        """
        self.last_http_attempts = 0
        self.last_finish_reason = ""
        self.last_prompt_tokens = None
        payload = self.prepare_chat_payload(system_prompt,user_prompt,images_base64)
        self.last_chat_payload_sha256 = self._payload_sha256(payload)
        self.last_chat_config_sha256 = self._chat_config_sha256()
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        }

        import time as _time
        _t0 = _time.monotonic()
        attempt = 0
        while True:
            if _CANCEL_FLAG["flag"]:
                raise VLMCancelledError("VLM call cancelled by SIGTERM (wall-clock timeout)")
            remaining = self.total_timeout_s - (_time.monotonic() - _t0)
            if remaining <= 0:
                return "__VLM_TIMEOUT__"
            try:
                self.last_http_attempts += 1
                record_http_attempt()
                resp = requests.post(self.api_url, json=payload, headers=headers,
                                     timeout=min(self.timeout, remaining))
                if resp.status_code in (401, 403):
                    logger.error(f"VLM API auth error {resp.status_code}: fail-fast")
                    return "__VLM_AUTH_ERROR__"
                if resp.status_code in (429, 500, 502, 503, 504):
                    base = self.retry_backoff_base * (1.5 if resp.status_code == 503 else 1.0)
                    if self._retry_wait(attempt, _t0, base):
                        attempt += 1
                        continue
                    try:
                        err_body = resp.text[:300]
                    except:
                        err_body = ""
                    logger.error(f"VLM API {resp.status_code} after {self.last_http_attempts} HTTP attempts (retry/time limit): {err_body}")
                    return "__VLM_HTTP_ERROR__"
                resp.raise_for_status()
                result = resp.json()
                usage=result.get('usage') or {}
                prompt_tokens=usage.get('prompt_tokens') if isinstance(usage,dict) else None
                self.last_prompt_tokens=prompt_tokens if type(prompt_tokens) is int and prompt_tokens>=0 else None
                # 2026-10-11 UGV six-view：只读元数据（finish_reason=length 截断
                # 须按协议错误处理，不靠补括号修复）。不改变任何返回行为。
                try:
                    self.last_finish_reason = str(
                        ((result.get("choices") or [{}])[0] or {}).get("finish_reason") or "")
                except Exception:
                    self.last_finish_reason = ""
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout as e:
                if self._retry_wait(attempt, _t0):
                    attempt += 1
                    continue
                logger.error(f"VLM API timeout after {self.last_http_attempts} HTTP attempts (retry/time limit): {e}")
                return "__VLM_TIMEOUT__"
            except requests.exceptions.HTTPError as e:
                # 400/404/422 等参数/端点错误：确定性错误（R2-P1-06），
                # 与鉴权同级 fatal——重试无意义，须快速失败让人工修配置，
                # 不能落 INFRA_ERROR 在 runner 里 continue 烧 VLM 预算。
                status_code = resp.status_code if resp is not None else None
                try:
                    err_body = resp.text[:500] if resp is not None else "no response"
                except:
                    err_body = "cannot read response"
                logger.error(f"VLM API HTTP error: {e} | status={status_code} | body={err_body}")
                if status_code in (400, 404, 422):
                    return "__VLM_CONFIG_ERROR__"
                return "__VLM_HTTP_ERROR__"
            except requests.exceptions.RequestException as e:
                if self._retry_wait(attempt, _t0):
                    attempt += 1
                    continue
                logger.error(f"VLM API request error after {self.last_http_attempts} HTTP attempts (retry/time limit): {e}")
                return "__VLM_HTTP_ERROR__"
            except VLMCancelledError:
                raise
            except Exception as e:
                logger.error(f"VLM API error: {e}")
                return "__VLM_HTTP_ERROR__"
        return "__VLM_HTTP_ERROR__"

    def decide(self, system_prompt, user_prompt, images_base64):
        """完整决策流程：调用 VLM → 解析 JSON → 返回 (parsed, status, raw)"""
        raw = self.chat(system_prompt, user_prompt, images_base64)
        if raw == "__VLM_TIMEOUT__":
            return None, "vlm_timeout", raw
        if raw == "__VLM_CONFIG_ERROR__":
            return None, "vlm_config_error", raw
        if raw == "__VLM_HTTP_ERROR__":
            return None, "vlm_http_error", raw
        if raw == "__VLM_AUTH_ERROR__":
            return None, "vlm_auth_error", raw
        if not raw:
            return None, "vlm_empty_response", ""
        parsed = repair_json(raw)
        if not isinstance(parsed, dict) or not parsed:
            return None, "json_parse_failed", raw
        if model_field_error(parsed):
            return None, "json_parse_failed", raw
        return parsed, "ok", raw
