# Tier 0 Fallback 설계

**작성일**: 2026-08-07
**상태**: 승인됨, 구현 대기

## 1. 목적

README 3절 아키텍처의 `Fallback (Tier0)` 상자를 **독립 실행 가능한 모듈**로 구현한다. 신경망 Mitigator와의 통합(에스컬레이션 사다리)은 이번 범위가 아니다.

### 왜 지금 필요한가

2026-08-07 기준, 이 프로젝트에는 **Detector 재검증을 통과하는 출력이 하나도 없다.**

| | 진폭 감소 | 판정 |
|---|---|---|
| 학습된 Mitigator (`best.pt`, val_psnr 35.66dB) | −48.5% | 위험 구간 잔존 |
| All-In-One-Deflicker (참고 구현) | −90~95% | 3개 중 2개 통과 |

Tier 0은 **반드시 통과하는 경로**를 확보한다. 강도를 올리면 화면이 단색에 수렴하고, 단색 프레임 사이에는 휘도 변화가 없으므로 검출기 기준을 반드시 만족한다.

### 안전망이자 계측기

Tier 0은 통과 보장 외에 다음을 **측정**한다. 이 값들은 후속 작업(PriorCalc 목표를 모션 보정 시간축 집계로 교체)의 설계 근거가 된다.

1. 클립·구간별 **최소 필요 강도 `s`** — "ITU 기준 통과에 대비를 얼마나 뭉개야 하는가"
2. `q75 → s` 관계 — 측정된 Δ 분포에서 필요 강도로 가는 변환
3. 그 강도에서의 **화질 손실 상한** — 신경망이 이겨야 하는 선. 현재 이 비교 기준이 없어 신경망 결과의 좋고 나쁨을 판단할 수 없다.
4. **목표 자체의 달성 가능성** — `s`가 0.9 근처로 나온다면 화면이 거의 단색이어야 통과한다는 뜻이고, 그것은 모델이 부족한 것이 아니라 검출 기준이 과도할 가능성을 시사한다. 이 가설은 아직 한 번도 검증된 적이 없다.

`s`는 **전역** 방식의 값이므로 픽셀별 방식의 목표치가 아니라 **상한선**이다. 픽셀별 방식은 같은 Δ 감소를 더 적은 손상으로 달성할 수 있어야 한다.

## 2. 알고리즘

### 2.1 변환식

각 프레임을 **선형 광에서 채널별로** 기준값 쪽으로 압축한다.

```
C_out = R_C + (C_in − R_C) · (1 − s)        C ∈ {R, G, B}, 선형 광
```

- `s = 0` → 항등변환
- `s = 1` → 프레임 전체가 기준색 단색

**감마 공간이 아니라 선형 광에서 하는 이유**: sRGB 값에 아핀 변환을 걸면 채널마다 비선형 곡선을 거치며 색이 틀어진다. 또한 아래 2.2의 `(1−s)` 성질이 성립하지 않는다.

**채널별로 같은 아핀을 거는 이유**: 상대휘도는 선형 광 채널의 가중합(`0.2126R + 0.7152G + 0.0722B`)이므로, 세 채널에 같은 아핀을 걸면 휘도에도 **정확히 같은 아핀**이 걸린다.

```
L_out = R_L + (L_in − R_L) · (1 − s)     where R_L = 0.2126·R_R + 0.7152·R_G + 0.0722·R_B
```

출력은 선형 광에서 `[0, 1]`로 클램프한 뒤 sRGB로 되돌린다.

### 2.2 강도를 탐색하지 않고 계산할 수 있는 이유

연속 두 프레임의 기준값이 같다면(`R_i ≈ R_{i+1}`), 픽셀별 휘도 차이는 정확히 `(1−s)`배가 된다.

```
L_out,i+1 − L_out,i = (1−s)(L_in,i+1 − L_in,i) + s(R_{i+1} − R_i)
                     ≈ (1−s)(L_in,i+1 − L_in,i)
```

기준값은 1초 trailing 평균이므로 프레임 하나 사이에는 거의 변하지 않는다. 따라서 근사가 성립한다.

`detector.flash.transition_mask`는 `delta > general_flash_delta_threshold`인 픽셀을 센다. 면적을 `max_area_ratio` 아래로 내리려면, **상위 `max_area_ratio` 분위의 Δ**가 임계 이하가 되어야 한다.

```
q = quantile(|Δ|, 1 − max_area_ratio)          # itu 프로파일: 75% 분위
s = clamp(1 − delta_threshold / q, 0, 1)       # q ≤ threshold이면 s = 0
```

**Δ는 반드시 모션 보정 후에 측정한다.** `detector.pipeline._iter_scores_and_masks`가 `compensate_shift(prev)` 대 `curr`로 비교하므로, 보정 없이 재면 카메라 팬이 Δ를 부풀려 필요 이상의 강도가 나온다.

한 구간의 `s`는 구간 내 모든 프레임 쌍에 대해 구한 값의 **최댓값**을 쓴다. 가장 심한 프레임 쌍이 기준을 넘으면 구간 전체가 위험으로 판정되기 때문이다.

### 2.3 계산값이 근사인 이유 — 그래서 재검증은 필수

세 가지 이유로 계산된 `s`는 출발점일 뿐이며, **통과 판정은 항상 실제 Detector가 한다.**

1. **`darker < dark_threshold` 조건**: `transition_mask`는 Δ뿐 아니라 두 프레임 중 어두운 쪽이 임계(0.8) 미만인지도 본다. 압축은 어두운 픽셀을 기준값 쪽으로 **밝히므로** 이 조건을 벗어나게 만들 수도, 반대로 밝은 픽셀을 낮춰 조건에 들어오게 만들 수도 있다.
2. **적색 플래시 마스크는 Δ가 아니다**: `red_flash_mask`는 `is_saturated_red(prev) != is_saturated_red(curr)`인 **불리언 XOR**이다. 압축은 채도를 기준색 쪽으로 낮추므로 `s → 1`에서는 모든 픽셀이 같은 색이 되어 XOR이 0이 되지만, 그 수렴은 **선형이 아니다.** 적색 플래시가 지배적인 구간에서는 계산값이 부족하게 나오고 상향 루프가 받아내야 한다.
3. **면적은 1초 윈도우 최댓값**: `max_flagged_area_in_window`는 프레임 쌍 단위 계산과 정확히 일치하지 않는다.

### 2.4 페이드

각 구간의 앞뒤 `margin_frames`(= `round(0.5 * fps)`, `scores_to_segments`가 쓰는 값과 동일) 구간에서 `s`를 raised cosine으로 0 ↔ 최대값 사이에서 올리고 내린다.

```
w(t) = 0.5 · (1 − cos(π · t))        t ∈ [0, 1], 마진 내 상대 위치
s_effective = s · w(t)
```

페이드가 없으면 구간 경계에서 밝기가 계단처럼 변하고, **그 급변 자체가 새로운 flicker로 검출될 수 있다.** 마진은 이미 위험 구간 바깥의 여유분이므로 페이드를 걸기에 적합하다.

구간 길이가 `2 * margin_frames`보다 짧으면 페이드 구간이 겹친다. 이 경우 각 페이드 길이를 `segment_length // 2`로 줄인다.

## 3. 모듈 구조

`detector/`, `prior/`, `mitigator/`와 나란히 `fallback/` 패키지를 만든다.

| 파일 | 책임 | 의존 |
|---|---|---|
| `fallback/transfer.py` | 대비 압축 변환. 프레임 하나 + 기준색 + 강도 → 프레임 하나 | `detector.luminance` |
| `fallback/strength.py` | 모션 보정 Δ 분포에서 필요 강도 계산 | `detector.luminance`, `detector.motion`, `detector.profiles` |
| `fallback/apply.py` | 구간별 적용 + 페이드 + 재검증/상향 루프 | 위 둘 + `detector.pipeline` |
| `fallback/cli.py` | 영상 → 영상, 리포트 JSON | `fallback.apply` |
| `fallback/tests/` | 테스트 | |

### 3.1 공개 인터페이스

```python
# fallback/transfer.py
def trailing_reference(frames_linear: list[np.ndarray], index: int, window_frames: int) -> np.ndarray:
    """index에서 끝나는 trailing 윈도우의 채널별 평균. shape (3,)."""

def compress_contrast(frame_rgb: np.ndarray, reference_linear: np.ndarray, strength: float) -> np.ndarray:
    """sRGB 프레임을 기준색 쪽으로 압축. 입출력 모두 sRGB [0,1] float32."""


# fallback/strength.py
def motion_compensated_deltas(frames: list[np.ndarray]) -> list[np.ndarray]:
    """연속 프레임 쌍의 모션 보정된 |휘도 차|. 길이 len(frames)-1."""

def required_strength(frames: list[np.ndarray], profile: ThresholdProfile) -> float:
    """이 프레임들의 면적을 profile.max_area_ratio 아래로 내리는 데 필요한 최소 강도."""


# fallback/apply.py
@dataclass
class SegmentOutcome:
    start_frame: int
    end_frame: int
    initial_strength: float      # 계산값
    final_strength: float        # 상향 후 최종값
    rounds: int
    passed: bool

@dataclass
class FallbackReport:
    segments: list[SegmentOutcome]
    passed: bool                 # 최종 전체 클립 재검증 결과
    rounds: int
    remaining_segments: int

def mitigate_with_fallback(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    max_rounds: int = 6,
    strength_step: float = 0.1,
) -> tuple[list[np.ndarray], FallbackReport]:
    """순수 함수: 프레임 리스트를 받아 보정된 프레임 리스트와 리포트를 반환."""
```

`mitigate_with_fallback`을 **프레임 리스트 in / 프레임 리스트 out의 순수 함수**로 두는 것은 의도적이다. 나중에 에스컬레이션 사다리에 통합할 때 이 함수를 호출만 하면 되도록 하기 위해서다.

### 3.2 기존 모듈 변경

`detector/luminance.py`에 sRGB ↔ 선형 광 변환을 **공개 함수로 노출**한다. 현재 `_srgb_to_linear`는 비공개이고 역변환은 없다.

```python
def srgb_to_linear(channel: np.ndarray) -> np.ndarray: ...   # 기존 _srgb_to_linear를 공개로
def linear_to_srgb(channel: np.ndarray) -> np.ndarray: ...   # 신규, 위의 정확한 역함수
```

`relative_luminance`는 새 `srgb_to_linear`를 호출하도록 바꾼다. 동작은 동일하며 기존 테스트가 그대로 통과해야 한다.

## 4. 재검증/상향 루프

```
s = {segment: required_strength(segment_frames, profile) for each segment}

for round in 1..max_rounds:
    out = 모든 구간에 s를 페이드 걸어 적용
    _, remaining = run_detection(out, fps, profile)        # 클립 전체
    if not remaining:
        return out, 통과
    남은 위험 프레임과 겹치는 구간만 s += strength_step (최대 1.0)

return out, 실패 (리포트에 잔존 구간 기록)
```

**클립 전체로 재검증하는 이유**: 구간만 떼어 검증하면 `WindowedFlashCounter`가 프레임 0을 `uncertain=True`로 무조건 flagged 처리하고 윈도우 프라이밍도 달라져, 실제 채점 조건과 다른 판정이 나온다. 최종 성공 기준은 클립 전체에 대한 판정이므로 루프도 같은 조건으로 돈다.

**겹치는 구간만 상향하는 이유**: 이미 통과한 구간까지 함께 뭉개면 "필요 이상으로 뭉개지 않는다"는 이 설계의 핵심 이점이 사라진다.

**비용**: `run_detection`은 프레임당 약 25ms이므로 931프레임 클립에서 라운드당 약 23초, 최대 6라운드로 약 2.3분. 오프라인 도구로서 수용 가능하다.

**`max_rounds` 소진 시**: 예외를 던지지 않고 `passed=False`인 리포트를 반환한다. 호출자(현재는 CLI)가 정책을 정한다. CLI는 이 경우 종료 코드 1을 반환하고 잔존 구간을 출력한다 — README 7절의 "인시던트 로깅"에 해당한다.

## 5. 테스트 전략

| 대상 | 검증 내용 |
|---|---|
| `compress_contrast` | `s=0`이면 입력과 동일 / `s=1`이면 모든 픽셀이 기준색 / 출력이 `[0,1]` 범위 |
| `compress_contrast` | **픽셀별 휘도 Δ가 정확히 `(1−s)`배** — 2.2의 근거가 되는 성질 |
| `compress_contrast` | 감마 왜곡 없음: 회색조 입력이 회색조로 유지 |
| `srgb_to_linear` / `linear_to_srgb` | 왕복 변환이 항등 (`atol=1e-6`) |
| `required_strength` | Δ 분포를 아는 합성 프레임 쌍에서 기대값이 나옴 |
| `required_strength` | 이미 안전한 클립에서 `0.0` 반환 |
| `motion_compensated_deltas` | 순수 평행이동 클립에서 Δ가 0에 가까움 (모션 보정이 실제로 동작) |
| 페이드 | 구간 경계에서 프레임 간 평균 휘도 변화가 `delta_threshold` 미만 |
| 짧은 구간 | 길이가 `2*margin_frames` 미만인 구간에서 페이드가 겹치지 않음 |
| **end-to-end (합성)** | 합성 flicker 클립이 fallback 후 **위험 구간 0개** |
| **end-to-end (실제)** | `test_mp4/원본.mp4`에서 peak 면적 < 0.25, 위험 구간 0개 |
| 리포트 | 상향이 일어난 구간의 `final_strength > initial_strength`, `rounds` 정확 |

마지막 두 개(굵게)가 이 작업의 실질적 성공 기준이다. 나머지는 그 두 개가 왜 성립하는지를 보장하는 단위 테스트다.

실제 영상 테스트는 CI에서 돌리기엔 무겁다(`원본.mp4` 931프레임, 재검증 포함 수 분). `@pytest.mark.slow`로 표시하고 기본 실행에서 제외하되, 구현 완료 시 반드시 수동으로 1회 실행한다.

## 6. 측정 산출물

구현 후 다음을 기록한다. 이것이 1절에서 말한 "계측기" 역할이다.

세 개의 실제 클립(`원본.mp4`, Anyma, Cera Khin)에 대해:

| 항목 | 의미 |
|---|---|
| 구간별 `initial_strength`, `final_strength` | 계산값이 얼마나 정확했는가 |
| 계산값과 최종값의 차이 | 2.3의 근사 오차 크기 |
| 통과 후 평균 휘도 변화율 | All-In-One-Deflicker의 −21~35%와 직접 비교 |
| 통과 후 프레임 내 대비(표준편차) 변화율 | 대비 압축의 실제 비용 |
| 최대 `final_strength` | 목표 달성 가능성 판단 (0.9 이상이면 기준 재검토 필요) |

## 7. 이번 범위에서 제외

- **에스컬레이션 사다리 통합** (README Level 0~3): 현재 신경망 경로가 재검증을 한 번도 통과하지 못하므로, 통합해도 항상 Level 1로 강등된다. 신경망이 가끔이라도 통과하기 시작한 뒤에 만든다.
- **BufferManager / 지연 예산**: 위와 같은 이유.
- **프레임 홀드 최후수단** (Level 3): `s=1.0`이 이미 강제 감광에 해당하고, 그것으로도 통과하지 못하는 사례를 아직 관측하지 못했다. 관측되면 그때 추가한다.
- **실시간 스트리밍 경로**: 알고리즘은 완전히 인과적(trailing 기준값, 미래 프레임 불필요)이므로 나중에 얹을 수 있다. 다만 재검증 루프는 구간 전체를 필요로 하므로 스트리밍에서는 강도 계산 방식이 달라져야 한다.
