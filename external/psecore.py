# -*- coding: utf-8 -*-
"""
psecore.py — 통합 광과민성 검출기 v2.0
================================================================================
팀에서 각자 만든 세 검출기를 하나로 합친다. 합치는 기준은 "누가 만들었나"가 아니라
**정답을 아는 코퍼스 14편에서 무엇이 맞았나**다. 실측 근거는 각 섹션 주석에 남긴다.

가져온 것
  · safeframe 에서 — 히스테리시스 peak-valley 전환 계수, **동기화 그룹핑**,
                     334ms 갭 예외, 적분영상 10° 창, LUT 휘도, 신뢰도 태그,
                     required_gain(완화 게인) 출력
  · detector 에서  — phase correlation 움직임 보상, **픽셀 마스크 export API**,
                     엄격한 스트리밍(인과적), 리포트에 한계를 박아넣는 caveats
  · pse_analyze 에서 — Michelson 분기, 색 대립축(RG/BY), 적청축(RB),
                     Δu'v' 게이트가 붙은 적색 규칙, 10° 국소면적 + 5초 누적(강화 모드)
  · **신규(RGB 채널)** — 아래 §0 참조

버린 것 (실측으로 틀린 것)
  · detector 의 flagged-frame 계수 — 정지화면조차 30/30 으로 포화해서
    "초당 3회" 규칙이 **항상 참**이 됐다. 판정이 면적 축 단독으로 결정되어
    합법적인 2.5Hz 전면 점멸이 FAIL 로 나왔다. peak-valley 쌍 계수로 대체.
  · detector 의 프레임 수 단언 — 컨테이너 선언 510 vs 실제 507 처럼 3프레임만
    어긋나도 예외를 던져 **실사 MP4 2편을 모두 읽기 거부**했다. 경고로 낮춘다.
  · safeframe 의 색 무시 — 미검출 3편이 전부 색 위험이었다(등휘도 적녹/청황/애니 급컷).
  · safeframe 의 움직임 보상 부재 — 흔들리기만 하는 안전 영상을 3.5초 FAIL.

--------------------------------------------------------------------------------
§0. 왜 RGB 채널을 새로 넣는가
--------------------------------------------------------------------------------
휘도는 Y = 0.2126R + 0.7152G + 0.0722B 다. 같은 100% 진폭 점멸이 휘도에 실리는 양:

    백색 ΔY 1.000 (임계 0.10 의 10.0배)      적색 ΔY 0.213 (2.1배)
    녹색 ΔY 0.715 (7.2배)                    **청색 ΔY 0.072 (0.72배 — 임계 미달)**

즉 **화면 전체가 검정↔순청으로 최대 진폭 점멸해도 휘도로는 검출되지 않는다.**
실측: "휘도는 임계 미달인데 어떤 채널은 초과"하는 화소가 Anyma 최대 8.4%,
콘서트 최대 60.7%.

임상 근거가 이를 뒷받침한다 — Parra 2007 단색 유발률은
적 88% / 청 72% / 백 68% / 녹 64% / 황 60% 로 **색깔에 거의 무관하게 평평**하다.
그런데 휘도 가중치는 10배 차이가 난다. 광과민성을 휘도로만 재는 것은
**임상적으로 틀린 근사**다. 그래서 채널별 선형 excursion 을 공통 임계로 본다.

주의: 이 채널은 표준에 없다. 기본 모드는 `warn`(리포트에만 표시)이고,
`--rgb fail` 로 켜야 FAIL 판정에 반영된다. 표준 준수 판정과 섞지 않기 위해서다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Literal, Sequence

import cv2
import numpy as np

__version__ = "2.0.0"

Mode = Literal["fail", "warn", "off"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 색공간 — LUT 기반 (safeframe 방식: (H,W,3) 중간 float 배열을 안 만든다)
# ══════════════════════════════════════════════════════════════════════════════

def _srgb_lut() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


_LIN = _srgb_lut()
_LUM_LUT = {
    "bt709":  tuple((_LIN * w).astype(np.float32) for w in (0.2126, 0.7152, 0.0722)),
    "bt2020": tuple((_LIN * w).astype(np.float32) for w in (0.2627, 0.6780, 0.0593)),
}

M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], np.float32)
# Smith-Pokorny LMS (L+M = Y 정규화)
M_XYZ2LMS = np.array([[0.15514,  0.54312, -0.03286],
                      [-0.15514, 0.45684,  0.03286],
                      [0.0,      0.0,      0.01608]], np.float32)
M_RGB2LMS = (M_XYZ2LMS @ M_RGB2XYZ).astype(np.float32)

UV_RED = (0.4507, 0.5229)      # BT.709 적 원색 u'v'
UV_BLUE = (0.1754, 0.1579)     # BT.709 청 원색 u'v'
_RBV = np.array([UV_BLUE[0] - UV_RED[0], UV_BLUE[1] - UV_RED[1]], np.float32)
_RB_DIR = (_RBV / np.linalg.norm(_RBV)).astype(np.float32)


def luminance(bgr_u8: np.ndarray, primaries: str = "bt709") -> np.ndarray:
    lr, lg, lb = _LUM_LUT.get(primaries, _LUM_LUT["bt709"])
    out = lr[bgr_u8[..., 2]]
    out += lg[bgr_u8[..., 1]]
    out += lb[bgr_u8[..., 0]]
    return out


def linear_rgb(bgr_u8: np.ndarray) -> np.ndarray:
    return _LIN[bgr_u8][..., ::-1]          # BGR LUT -> RGB 순서


def uv_prime(lin_rgb: np.ndarray) -> np.ndarray:
    """CIE 1976 UCS (u', v'). 지침이 색 전환 크기를 재는 좌표계."""
    XYZ = lin_rgb @ M_RGB2XYZ.T
    d = np.maximum(XYZ[..., 0] + 15.0 * XYZ[..., 1] + 3.0 * XYZ[..., 2], 1e-6)
    return np.stack([4.0 * XYZ[..., 0] / d, 9.0 * XYZ[..., 1] / d], -1)


def cone_contrast(lin_rgb: np.ndarray, bg_lms: np.ndarray):
    """원추 대비 공간에서 RG(L−M) · BY(S−(L+M)) 축.

    **절대 LMS 로 하면 안 된다** — L_white ≠ M_white 라서 무채색 스트로브가
    가짜 적녹 신호를 만든다(실측 허위 위반 0.57초). 대비 공간에서 분해한다.
    """
    lms = lin_rgb @ M_RGB2LMS.T
    cc = lms / np.maximum(bg_lms, 1e-4) - 1.0
    rg = cc[..., 0] - cc[..., 1]
    by = cc[..., 2] - 0.5 * (cc[..., 0] + cc[..., 1])
    return lms, rg, by


# ══════════════════════════════════════════════════════════════════════════════
# 2. 전환 검출 — 히스테리시스 peak-valley  (safeframe 방식)
# ══════════════════════════════════════════════════════════════════════════════

class PeakValley:
    """
    n프레임 룩백 창 안의 극값 대비 변화로 전환을 등록한다.

    두 가지를 동시에 해결한다.
      · **같은 방향 연속 진행을 1회로 누적** — 프레임 차분으로 세면 완만한 상승이
        여러 회로 쪼개진다.
      · **느린 페이드 배제** — 룩백 창(T_qualify) 안에서 임계를 못 넘으면 전환이
        아니다. 표준의 "전환은 짧아야 한다"에 대응.

    극성이 반드시 교번하므로 **상승 전환 수 = 플래시(반대 방향 한 쌍) 수**다.
    상승에 고정하는 이유: 하강부터 세기 시작하면 위상이 반 주기 어긋나
    동기화 판정이 무너진다.
    """

    def __init__(self, shape, n_lookback: int, theta: float):
        self.n = max(1, int(n_lookback))
        self.theta = float(theta)
        self.last_pol = np.zeros(shape, np.int8)
        self._ring: list[np.ndarray] = []

    def step(self, X: np.ndarray, qualify=None):
        """qualify(hi, lo) -> bool 마스크. 임계 외 추가 조건(어두운쪽 상한 등)."""
        if not self._ring:
            self._ring.append(X.copy())
            z = np.zeros(X.shape, bool)
            return z, np.zeros(X.shape, np.float32), z.copy()

        lmax = self._ring[0].copy()
        lmin = self._ring[0].copy()
        for r in self._ring[1:]:
            np.maximum(lmax, r, out=lmax)
            np.minimum(lmin, r, out=lmin)

        d_down = lmax - X
        d_up = X - lmin
        down = d_down >= self.theta
        up = d_up >= self.theta
        if qualify is not None:
            down &= qualify(lmax, X)        # 하강: 밝은쪽=lmax, 어두운쪽=X
            up &= qualify(X, lmin)          # 상승: 밝은쪽=X,    어두운쪽=lmin

        down &= (self.last_pol != -1)
        up &= (self.last_pol != 1)
        both = down & up
        if both.any():
            pd = d_down >= d_up
            down &= ~(both & ~pd)
            up &= ~(both & pd)

        delta = np.zeros(X.shape, np.float32)
        np.copyto(delta, d_down, where=down)
        np.copyto(delta, d_up, where=up)
        self.last_pol = np.where(down, np.int8(-1),
                                 np.where(up, np.int8(1), self.last_pol)).astype(np.int8)

        self._ring.append(X.copy())
        if len(self._ring) > self.n:
            self._ring.pop(0)
        return up, delta, (down | up)


class BinaryTransition:
    """이진 상태(포화 적색 여부 등)의 전환. 게이트 조건을 함께 받는다."""

    def __init__(self, shape):
        self.last_pol = np.zeros(shape, np.int8)
        self.prev_state = None

    def step(self, state: np.ndarray, gate: np.ndarray, magnitude: np.ndarray):
        if self.prev_state is None:
            self.prev_state = state.copy()
            z = np.zeros(state.shape, bool)
            return z, np.zeros(state.shape, np.float32), z.copy()
        enter = gate & state & ~self.prev_state
        leave = gate & ~state & self.prev_state
        enter &= (self.last_pol != 1)
        leave &= (self.last_pol != -1)
        delta = np.zeros(state.shape, np.float32)
        np.copyto(delta, magnitude, where=(enter | leave))
        self.last_pol = np.where(enter, np.int8(1),
                                 np.where(leave, np.int8(-1), self.last_pol)).astype(np.int8)
        chg = enter | leave
        np.copyto(self.prev_state, state, where=chg)
        return enter, delta, chg


# ══════════════════════════════════════════════════════════════════════════════
# 3. 픽셀별 플래시 카운터 — 증분 갱신 + 334ms 갭 예외  (safeframe 방식)
# ══════════════════════════════════════════════════════════════════════════════

class FlashCounter:
    NEVER = -1e9

    def __init__(self, shape, window_frames: int, gap_ms: float):
        self.wf = max(1, int(window_frames))
        self.gap_ms = float(gap_ms)
        self.win = np.zeros(shape, np.int16)
        self.seq = np.zeros(shape, np.int16)
        self.last_t = np.full(shape, self.NEVER, np.float32)
        self._ring: list[np.ndarray] = []
        self._total = 0

    def push(self, flash: np.ndarray, t_ms: float) -> None:
        n = int(flash.sum())
        if n:
            cont = flash & ((t_ms - self.last_t) <= self.gap_ms)
            self.seq[flash & ~cont] = 1
            self.seq[cont] += 1
            np.copyto(self.last_t, np.float32(t_ms), where=flash)
        if self._total or n:
            self.seq[(t_ms - self.last_t) > self.gap_ms] = 0
        f8 = flash.view(np.int8)
        self.win += f8
        self._total += n
        self._ring.append(f8)
        if len(self._ring) > self.wf:
            old = self._ring.pop(0)
            self.win -= old
            self._total -= int(old.sum())

    @property
    def has_any(self) -> bool:
        return self._total > 0

    def counts(self, use_gap: bool) -> np.ndarray:
        return np.minimum(self.win, self.seq) if use_gap else self.win


# ══════════════════════════════════════════════════════════════════════════════
# 4. 면적 — 전역 / WCAG 10° 창 / 동기화 그룹핑  (safeframe 방식)
# ══════════════════════════════════════════════════════════════════════════════

def wcag_window_px(w: int, h: int, field_deg: float, fov_deg: float) -> int:
    return max(1, min(int(round(w * field_deg / max(fov_deg, 1e-6))), w, h))


def area_global(B: np.ndarray) -> float:
    return float(B.sum()) / float(B.size)


def area_wcag(B: np.ndarray, w: int):
    """모든 w×w 창 중 최대 점유율. 적분영상으로 O(N)."""
    H, W = B.shape
    w = max(1, min(w, H, W))
    S = cv2.integral(B.astype(np.uint8), sdepth=cv2.CV_32S)
    box = S[w:, w:] - S[:-w, w:] - S[w:, :-w] + S[:-w, :-w]
    i = int(np.argmax(box))
    y, x = divmod(i, box.shape[1])
    return float(box.flat[i]) / float(w * w), (int(x), int(y))


def sync_groups(last_t: np.ndarray, over: np.ndarray, tol_ms: float, frame_ms: float):
    """
    **동기화 그룹핑** — 표준은 "같은 영역이 동시에" 점멸할 때만 면적을 합산한다.
    위상이 다른 두 영역을 더하면 없는 위험을 만든다. 셋 중 safeframe 만 갖고 있던 기능.
    """
    vals = last_t[over]
    if vals.size == 0:
        return []
    fi = np.rint(vals / frame_ms).astype(np.int64)
    base = int(fi.min())
    occ = np.nonzero(np.bincount(fi - base))[0]
    tol_f = max(1, int(tol_ms / frame_ms))
    groups = [[int(occ[0])]]
    for o in occ[1:]:
        if int(o) - groups[-1][-1] <= tol_f:
            groups[-1].append(int(o))
        else:
            groups.append([int(o)])
    return [((g[0] + base) * frame_ms, (g[-1] + base) * frame_ms) for g in groups]


# ══════════════════════════════════════════════════════════════════════════════
# 5. 움직임 보상  (detector 방식)
# ══════════════════════════════════════════════════════════════════════════════

def global_shift(prev_gray: np.ndarray, curr_gray: np.ndarray):
    try:
        (dx, dy), _ = cv2.phaseCorrelate(prev_gray.astype(np.float32),
                                         curr_gray.astype(np.float32))
    except cv2.error:
        return 0.0, 0.0
    return float(dx), float(dy)


def warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 프로파일 — 신뢰도 태그 포함  (safeframe 방식)
#    [확정] 표준 원문 검증  [임상] 임상 문헌 근거  [미검증] 확인 필요  [미규정] 표준 밖
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Cfg:
    # ── 강도 ──────────────────────────────────────────────────────────────
    # [확정] 상대휘도 0.10 == 20 cd/m² @ L_peak 200. **비율로 쓴다** —
    #        20/160 은 기준 200 일 때의 예시일 뿐이다(Jordan & Vanderheiden 2024).
    theta_lum: float = 0.10
    theta_dark_max: float = 0.80
    michelson: float = 1.0 / 17.0      # [확정] 어두운쪽이 상한 이상일 때의 분기
    # [확정] 채도 적색 R/(R+G+B) ≥ 0.8 + [확정] Δu'v' ≥ 0.2
    red_ratio: float = 0.80
    red_min_v: float = 0.25
    theta_uv: float = 0.20
    # [임상] Parra 2007 — 적청 교대 유발률 100%(최고). u'v' 평면 대각선이라
    #        DKL 기본축에 나뉘어 들어가므로 전용 축으로 분리한다.
    rb_cos_min: float = 0.90
    # [미검증] 원추 대비 임계 — 임상 근거 없음. 추정치.
    theta_rg: float = 0.10
    theta_by: float = 0.20
    dark_gate_y: float = 0.004
    adapt_tau_s: float = 0.25
    # [미규정/신규] 채널별 RGB excursion — §0 참조. 기본 warn.
    theta_rgb: float = 0.10
    rgb_mode: Mode = "warn"
    red_mode: Mode = "fail"
    chroma_mode: Mode = "fail"          # RG/BY/RB

    # ── 빈도 ──────────────────────────────────────────────────────────────
    max_flashes_per_sec: float = 3.0    # [확정]
    window_ms: float = 1000.0
    gap_exception_ms: float = 334.0     # [확정] ITU-R/Ofcom
    use_gap_exception: bool = False
    T_qualify_ms: float = 50.0          # [미규정] 전환의 유효 지속
    T_sync_ms: float = 20.0             # [미규정] 동기화 허용치

    # ── 면적 ──────────────────────────────────────────────────────────────
    area_mode: Literal["global", "wcag"] = "global"
    area_ratio: float = 0.25            # [확정]
    wcag_field_deg: float = 8.86        # 10° 지름 원과 등면적인 정사각형 한 변
    fov_h_deg: float = 22.0             # [미규정] 데스크톱 기준

    # ── 강화(임상) 모드 ───────────────────────────────────────────────────
    strict: bool = False                # 10° 국소면적 + 5초 누적
    cumulative_sec: float = 5.0

    # ── 실행 ──────────────────────────────────────────────────────────────
    short_side: int = 240               # 분석 해상도(짧은 변)
    motion_comp: bool = True            # detector 의 흔들림 오탐 방지
    primaries: str = "bt709"


PROFILES: dict[str, Cfg] = {
    "bt1702":  Cfg(use_gap_exception=True, area_mode="global"),
    "ofcom":   Cfg(use_gap_exception=True, area_mode="global"),
    "iso9241": Cfg(use_gap_exception=False, area_mode="global"),
    "wcag":    Cfg(use_gap_exception=False, area_mode="wcag"),
    "strict":  Cfg(use_gap_exception=False, area_mode="wcag", strict=True,
                   rgb_mode="fail"),
}

CHANNELS = ("LUM", "RGB", "RED", "RG", "BY", "RB")
CH_LABEL = {"LUM": "휘도", "RGB": "RGB 채널별", "RED": "채도 적색",
            "RG": "적녹 대립", "BY": "청황 대립", "RB": "적청 교대"}

CAVEATS = [
    "이 도구는 콘텐츠가 안전하다고 보증하지 않는다. 상용 배포 전 Harding FPA 등 "
    "외부 검증이 필요하다.",
    "RG/BY 임계값은 임상 근거가 없는 추정치다. RB 축만 Parra 2007 근거가 있다.",
    "RGB 채널별 축은 표준에 없는 신규 축이다. 기본 모드는 warn 이며 FAIL 판정에 "
    "반영되지 않는다.",
    "전환 지속시간 66ms 조항, 20ms 면적 합산 동기화, 416x416 px 면적 규칙은 "
    "미구현이다.",
    "움직임 보상은 전역(팬) 한정이다. 국소 물체 움직임은 보상되지 않는다.",
]


# ══════════════════════════════════════════════════════════════════════════════
# 7. 비디오 리더 — detector 의 치명적 버그를 고친 판
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoInfo:
    path: str
    fps: float
    frames_declared: int
    src_w: int
    src_h: int
    ana_w: int
    ana_h: int


def open_video(path: str, short_side: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(w, h) <= short_side:
        aw, ah = w, h
    else:
        s = short_side / min(w, h)
        aw, ah = max(2, int(round(w * s))), max(2, int(round(h * s)))
    return cap, VideoInfo(path, fps, n, w, h, aw, ah)


def iter_frames(cap, info: VideoInfo) -> Iterator[np.ndarray]:
    """
    **detector 의 프레임 수 단언을 제거했다.** 컨테이너 선언과 실제 디코드가
    3프레임만 어긋나도 예외를 던져 실사 MP4 2편을 통째로 거부했다(실측).
    MP4 에서 흔한 일이므로 경고로 낮추고 디코드된 만큼 처리한다.
    """
    need = (info.ana_w, info.ana_h) != (info.src_w, info.src_h)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if need:
            f = cv2.resize(f, (info.ana_w, info.ana_h), interpolation=cv2.INTER_AREA)
        yield f


# ══════════════════════════════════════════════════════════════════════════════
# 8. 리포트
# ══════════════════════════════════════════════════════════════════════════════

def _tc(ms: float) -> str:
    ms = max(0.0, ms)
    h, r = divmod(int(ms), 3_600_000)
    m, r = divmod(r, 60_000)
    s, r = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{r:03d}"


@dataclass
class Segment:
    channel: str
    start_ms: float
    end_ms: float
    max_count: int = 0
    max_area: float = 0.0
    max_delta: float = 0.0
    window_origin: tuple | None = None

    def to_dict(self, cfg: Cfg):
        # 완화 게인: 이 구간의 최대 변화폭을 임계 아래로 낮추려면 몇 배로 눌러야 하나
        thr = cfg.theta_lum if self.channel in ("LUM", "RGB") else cfg.theta_uv
        gain = min(1.0, (thr * 0.9) / max(self.max_delta, 1e-6))
        d = {
            "channel": self.channel,
            "label": CH_LABEL.get(self.channel, self.channel),
            "start": _tc(self.start_ms), "end": _tc(self.end_ms),
            "duration_ms": round(self.end_ms - self.start_ms, 1),
            "measured": {"flashes_per_sec": self.max_count,
                         "area_ratio": round(self.max_area, 4),
                         "delta": round(self.max_delta, 4)},
            "threshold": {"flashes_per_sec": cfg.max_flashes_per_sec,
                          "area_ratio": cfg.area_ratio, "delta": thr},
            "exceedance": {
                "rate": round(self.max_count / max(cfg.max_flashes_per_sec, 1e-9), 2),
                "area": round(self.max_area / max(cfg.area_ratio, 1e-9), 2)},
            "remediation": {"required_gain": round(gain, 4)},
        }
        if self.window_origin:
            d["measured"]["window_origin"] = list(self.window_origin)
        return d


@dataclass
class Report:
    source: str
    profile: str
    info: dict
    params: dict
    segments: list = field(default_factory=list)
    warn_segments: list = field(default_factory=list)
    elapsed_sec: float = 0.0
    decode_note: str = ""

    @property
    def verdict(self) -> str:
        return "FAIL" if self.segments else "PASS"

    def channel_seconds(self) -> dict:
        out = {c: 0.0 for c in CHANNELS}
        for s in self.segments + self.warn_segments:
            out[s.channel] = out.get(s.channel, 0.0) + (s.end_ms - s.start_ms) / 1000.0
        return {k: round(v, 2) for k, v in out.items()}

    def to_dict(self, cfg: Cfg) -> dict:
        iv = sorted((s.start_ms, s.end_ms) for s in self.segments)
        merged: list[list[float]] = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        total = self.info.get("duration_ms", 0.0)
        viol = min(sum(b - a for a, b in merged), total) if total else 0.0
        return {
            "tool": f"psecore {__version__}",
            "source": self.source, "profile": self.profile,
            "verdict": self.verdict,
            "failed_channels": sorted({s.channel for s in self.segments}),
            "video": self.info, "params": self.params,
            "violations": [s.to_dict(cfg) for s in self.segments],
            "warnings": [s.to_dict(cfg) for s in self.warn_segments],
            "channel_seconds": self.channel_seconds(),
            "summary": {"total_ms": round(total, 1),
                        "violating_ms": round(viol, 1),
                        "violating_ratio": round(viol / total, 5) if total else 0.0,
                        "segments": len(merged),
                        "elapsed_sec": round(self.elapsed_sec, 2)},
            "decode_note": self.decode_note,
            "caveats": CAVEATS,
        }

    def to_text(self, cfg: Cfg) -> str:
        d = self.to_dict(cfg)
        mark = "X FAIL" if self.verdict == "FAIL" else "O PASS"
        L = ["=" * 72,
             f" {mark}   [{self.profile}]  {self.source}",
             "=" * 72,
             f" {self.info['src_w']}x{self.info['src_h']} -> 분석 "
             f"{self.info['ana_w']}x{self.info['ana_h']} | {self.info['fps']:.2f}fps"
             f" | {_tc(self.info['duration_ms'])} | {self.elapsed_sec:.1f}s 소요"]
        if self.decode_note:
            L.append(f" ! {self.decode_note}")
        L.append("")
        L.append(f" {'채널':<12}{'판정':>8}{'위반(초)':>10}   비고")
        L.append(" " + "-" * 62)
        cs = self.channel_seconds()
        failed = {s.channel for s in self.segments}
        warned = {s.channel for s in self.warn_segments}
        for c in CHANNELS:
            st = "FAIL" if c in failed else ("WARN" if c in warned else "PASS")
            note = ""
            if c == "RGB" and cfg.rgb_mode != "fail":
                note = "표준 밖 · warn 전용"
            if c in ("RG", "BY"):
                note = "임계 임상근거 없음"
            L.append(f" {CH_LABEL[c]:<12}{st:>8}{cs.get(c, 0.0):>10.2f}   {note}")
        if self.segments:
            L.append("")
            L.append(" 위반 구간")
            for s in self.segments[:12]:
                sd = s.to_dict(cfg)
                L.append(f"   {sd['start']} ~ {sd['end']}  {CH_LABEL[s.channel]:<10}"
                         f" {s.max_count}회/s (한도 {cfg.max_flashes_per_sec:.0f})"
                         f"  면적 {s.max_area*100:.0f}%"
                         f"  완화게인 {sd['remediation']['required_gain']:.3f}")
            if len(self.segments) > 12:
                L.append(f"   ... 외 {len(self.segments)-12}개")
        L.append("=" * 72)
        return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
# 9. 엔진
# ══════════════════════════════════════════════════════════════════════════════

class _ChannelState:
    def __init__(self, name, shape, cfg: Cfg, win_frames, n_look, theta, binary=False):
        self.name = name
        self.machine = (BinaryTransition(shape) if binary
                        else PeakValley(shape, n_look, theta))
        self.counter = FlashCounter(shape, win_frames, cfg.gap_exception_ms)
        self.run = None          # 진행 중인 위반 구간
        self.warn_run = None


def _accumulate(state, out_list, active, t_ms, ch, cnt, area, delta, wxy, force=False):
    if active and not force:
        if state is None:
            state = Segment(ch, t_ms, t_ms, cnt, area, delta, wxy)
        else:
            state.end_ms = t_ms
            state.max_count = max(state.max_count, cnt)
            state.max_area = max(state.max_area, area)
            state.max_delta = max(state.max_delta, delta)
            if wxy and state.window_origin is None:
                state.window_origin = wxy
    else:
        if state is not None:
            state.end_ms = t_ms
            out_list.append(state)
            state = None
    return state


def analyze(path: str, cfg: Cfg = None, profile_name: str = "custom",
            want_masks: bool = False, progress=None) -> Report:
    """
    영상 1편 분석. 스트리밍(인과적)이라 긴 영상도 메모리에 올리지 않는다.
    want_masks=True 면 프레임별 채널 마스크를 함께 반환한다 —
    **CNN/후처리 연동용** (detector 의 run_detection_with_masks 패턴).
    이 경우 O(N·H·W) 메모리를 쓰므로 짧은 클립에만 쓸 것.
    """
    cfg = cfg or PROFILES["bt1702"]
    t0 = time.time()
    cap, info = open_video(path, cfg.short_side)
    fps = info.fps
    frame_ms = 1000.0 / fps
    shape = (info.ana_h, info.ana_w)
    win_frames = max(1, int(round(cfg.window_ms / frame_ms)))
    n_look = max(1, int(round(cfg.T_qualify_ms / frame_ms)))
    tol_ms = max(cfg.T_sync_ms, 1.5 * frame_ms)
    win_px = wcag_window_px(info.ana_w, info.ana_h, cfg.wcag_field_deg, cfg.fov_h_deg)
    alpha = 1.0 - float(np.exp(-1.0 / (fps * cfg.adapt_tau_s)))

    S = {
        "LUM": _ChannelState("LUM", shape, cfg, win_frames, n_look, cfg.theta_lum),
        "RGB": _ChannelState("RGB", shape, cfg, win_frames, n_look, cfg.theta_rgb),
        "RED": _ChannelState("RED", shape, cfg, win_frames, n_look, 0.0, binary=True),
        "RG":  _ChannelState("RG",  shape, cfg, win_frames, n_look, cfg.theta_rg),
        "BY":  _ChannelState("BY",  shape, cfg, win_frames, n_look, cfg.theta_by),
        "RB":  _ChannelState("RB",  shape, cfg, win_frames, n_look, cfg.theta_uv),
    }
    mode_of = {"LUM": "fail", "RGB": cfg.rgb_mode, "RED": cfg.red_mode,
               "RG": cfg.chroma_mode, "BY": cfg.chroma_mode, "RB": cfg.chroma_mode}

    viol: list[Segment] = []
    warns: list[Segment] = []
    masks: dict[str, list] = {c: [] for c in CHANNELS} if want_masks else None
    bg_lms = None
    prev_gray = None
    n = 0

    def lum_qualify(hi, lo):
        # [확정] 어두운쪽 < 상한이면 절대차, 그 이상이면 Michelson 분기
        mich = (hi - lo) / np.maximum(hi + lo, 1e-6)
        return (lo < cfg.theta_dark_max) | (mich > cfg.michelson)

    try:
        for idx, bgr in enumerate(iter_frames(cap, info)):
            t_ms = idx * frame_ms
            n = idx + 1

            # ---- 움직임 보상 (detector) : 흔들림 오탐 방지
            # 위상상관용 그레이는 **이미 계산한 휘도를 재사용**한다(cvtColor 1회 절감).
            # 실제 warp 이 필요한 프레임에서만 휘도를 다시 계산한다.
            Lum = luminance(bgr, cfg.primaries)
            if cfg.motion_comp and prev_gray is not None:
                dx, dy = global_shift(prev_gray, Lum)
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    bgr = warp(bgr, -dx, -dy)
                    Lum = luminance(bgr, cfg.primaries)
            prev_gray = Lum
            lin = linear_rgb(bgr)
            lit = Lum > cfg.dark_gate_y

            # ---- LUM
            f, d, _ = S["LUM"].machine.step(Lum, qualify=lum_qualify)
            S["LUM"].counter.push(f, t_ms)
            dmax = {"LUM": float(d.max()) if d.size else 0.0}
            if want_masks:
                masks["LUM"].append(f)

            # ---- RGB 채널별 (§0) : 세 채널 중 가장 큰 excursion
            #      청색 점멸이 휘도에 안 실리는 문제를 정면으로 잡는 축.
            chmax = lin.max(axis=2)
            f, d, _ = S["RGB"].machine.step(chmax)
            S["RGB"].counter.push(f, t_ms)
            dmax["RGB"] = float(d.max()) if d.size else 0.0
            if want_masks:
                masks["RGB"].append(f)

            # ---- 색도 기반 축들
            uv = uv_prime(lin)
            tot = lin.sum(2) + 1e-6
            is_red = (lin[..., 0] / tot >= cfg.red_ratio) & (lin[..., 0] >= cfg.red_min_v)
            if not hasattr(S["RED"], "_prev_uv"):
                S["RED"]._prev_uv = uv.copy()
            duv = np.linalg.norm(uv - S["RED"]._prev_uv, axis=-1)
            # [확정] 적색 전환은 채도 조건 + Δu'v' ≥ 0.2 를 함께 요구한다
            f, d, _ = S["RED"].machine.step(is_red, duv >= cfg.theta_uv, duv)
            S["RED"].counter.push(f, t_ms)
            dmax["RED"] = float(d.max()) if d.size else 0.0
            if want_masks:
                masks["RED"].append(f)

            # ---- RB : 적청 축 투영 (Parra 2007)
            proj = (uv[..., 0] * _RB_DIR[0] + uv[..., 1] * _RB_DIR[1]).astype(np.float32)
            f, d, _ = S["RB"].machine.step(proj)
            S["RB"].counter.push(f & lit, t_ms)
            dmax["RB"] = float(d.max()) if d.size else 0.0
            if want_masks:
                masks["RB"].append(f & lit)
            S["RED"]._prev_uv = uv

            # ---- RG / BY : 원추 대비
            lms = lin @ M_RGB2LMS.T
            bg_lms = lms.copy() if bg_lms is None else (alpha * lms + (1 - alpha) * bg_lms)
            cc = lms / np.maximum(bg_lms, 1e-4) - 1.0
            rg = (cc[..., 0] - cc[..., 1]).astype(np.float32)
            by = (cc[..., 2] - 0.5 * (cc[..., 0] + cc[..., 1])).astype(np.float32)
            for key, sig in (("RG", rg), ("BY", by)):
                f, d, _ = S[key].machine.step(sig)
                S[key].counter.push(f & lit, t_ms)
                dmax[key] = float(d.max()) if d.size else 0.0
                if want_masks:
                    masks[key].append(f & lit)

            # ---- 채널별 판정
            for ch in CHANNELS:
                st = S[ch]
                if mode_of[ch] == "off" or not st.counter.has_any:
                    st.run = _accumulate(st.run, viol, False, t_ms, ch, 0, 0, 0, None)
                    st.warn_run = _accumulate(st.warn_run, warns, False, t_ms, ch,
                                              0, 0, 0, None)
                    continue
                cnt = st.counter.counts(cfg.use_gap_exception)
                mc = int(cnt.max())
                over = cnt > cfg.max_flashes_per_sec
                haz, best, bxy = False, 0.0, None
                if over.any():
                    for lo, hi in sync_groups(st.counter.last_t, over, tol_ms, frame_ms):
                        B = over & (st.counter.last_t >= lo - 0.5 * frame_ms) \
                                 & (st.counter.last_t <= hi + 0.5 * frame_ms)
                        if not B.any():
                            continue
                        if cfg.strict or cfg.area_mode == "wcag":
                            r, xy = area_wcag(B, win_px)
                        else:
                            r, xy = area_global(B), None
                        if r > best:
                            best, bxy = r, xy
                        if r > cfg.area_ratio:
                            haz = True
                is_fail = haz and mode_of[ch] == "fail"
                is_warn = haz and mode_of[ch] == "warn"
                st.run = _accumulate(st.run, viol, is_fail, t_ms, ch, mc, best,
                                     dmax[ch], bxy)
                st.warn_run = _accumulate(st.warn_run, warns, is_warn, t_ms, ch, mc,
                                          best, dmax[ch], bxy)

            if progress and idx % 200 == 0:
                progress(idx, info.frames_declared)
    finally:
        cap.release()

    end_t = n * frame_ms
    for ch in CHANNELS:
        S[ch].run = _accumulate(S[ch].run, viol, False, end_t, ch, 0, 0, 0, None, True)
        S[ch].warn_run = _accumulate(S[ch].warn_run, warns, False, end_t, ch,
                                     0, 0, 0, None, True)

    note = ""
    if info.frames_declared and abs(info.frames_declared - n) > 0:
        note = (f"컨테이너 선언 {info.frames_declared}프레임 / 실제 디코드 {n}프레임 "
                f"— 디코드된 만큼 처리했습니다(MP4 에서 흔한 차이).")

    rep = Report(
        source=path, profile=profile_name,
        info={"src_w": info.src_w, "src_h": info.src_h,
              "ana_w": info.ana_w, "ana_h": info.ana_h,
              "fps": round(fps, 3), "frames": n,
              "duration_ms": round(n * frame_ms, 1)},
        params={"T_qualify_ms": cfg.T_qualify_ms, "T_qualify_frames": n_look,
                "T_sync_ms": cfg.T_sync_ms, "T_sync_effective_ms": round(tol_ms, 2),
                "window_frames": win_frames, "wcag_window_px": win_px,
                "area_mode": "wcag" if (cfg.strict or cfg.area_mode == "wcag") else "global",
                "motion_comp": cfg.motion_comp, "strict": cfg.strict,
                "modes": mode_of},
        segments=sorted(viol, key=lambda s: (s.start_ms, s.channel)),
        warn_segments=sorted(warns, key=lambda s: (s.start_ms, s.channel)),
        elapsed_sec=time.time() - t0, decode_note=note)
    if want_masks:
        return rep, masks
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=f"psecore {__version__} — 통합 광과민성 검출기")
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("-p", "--profile", default="bt1702", choices=list(PROFILES))
    ap.add_argument("--rgb", default=None, choices=["fail", "warn", "off"],
                    help="RGB 채널별 축의 동작 (기본: 프로파일 값)")
    ap.add_argument("--no-motion", action="store_true", help="움직임 보상 끄기")
    ap.add_argument("--fov", type=float, default=None, help="수평 시야각(도)")
    ap.add_argument("--short-side", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    base = PROFILES[a.profile]
    kw = {}
    if a.rgb:
        kw["rgb_mode"] = a.rgb
    if a.no_motion:
        kw["motion_comp"] = False
    if a.fov:
        kw["fov_h_deg"] = a.fov
    if a.short_side:
        kw["short_side"] = a.short_side
    cfg = Cfg(**{**asdict(base), **kw}) if kw else base

    out, rc = [], 0
    for src in a.srcs:
        try:
            rep = analyze(src, cfg, profile_name=a.profile)
        except Exception as e:
            print(f"오류 {src}: {e}", file=sys.stderr)
            rc = max(rc, 2)
            continue
        if not a.quiet:
            print(rep.to_text(cfg))
        out.append(rep.to_dict(cfg))
        if rep.verdict == "FAIL":
            rc = max(rc, 1)
    if a.json:
        json.dump(out if len(out) != 1 else out[0],
                  open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
