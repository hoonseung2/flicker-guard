# psecore 검출기 어댑터 — 설계 문서

## 1. 배경과 목적

팀에서 세 개의 검출기(safeframe, detector, pse_analyze)를 하나로 합친 통합 검출기 `psecore.py`를 만들었고, 정답을 아는 코퍼스 14편으로 검증했다. 팀 전체가 같은 판정 기준을 쓰기 위해 flicker-guard도 이 검출기를 채택한다.

`external/psecore.py`에 스테이징되어 있으나 현재 어떤 코드도 이를 import하지 않는다.

**목표는 "판정 결과 일치"다.** 같은 영상을 넣으면 팀과 같은 위험 구간이 나오면 된다. 코드 파일 자체를 공유할 필요는 없다 (2026-08-05 확인).

### psecore가 우리 검출기보다 낫다고 주장하는 점

- **peak-valley 쌍 계수** — 우리의 flagged-frame 계수는 실제 플래시 횟수를 약 2배로 부풀린다(우리 코드 주석에도 명시된 알려진 한계). 합법적인 2.5Hz 점멸이 FAIL로 나올 수 있다.
- **RGB 채널별 축** — 청색은 휘도 가중치가 0.0722라, 화면 전체가 검정↔순청으로 최대 진폭 점멸해도 휘도 기반 검출기는 원리적으로 못 잡는다. 팀 실측: 콘서트 영상에서 최대 60.7% 화소가 "휘도는 임계 미달인데 어떤 채널은 초과".
- **동기화 그룹핑** — 위상이 다른 두 영역의 면적을 합산하지 않는다.
- **프레임 수 불일치 허용** — 컨테이너 선언과 실제 디코드가 어긋나도 경고로 처리. 우리도 2026-08-04에 같은 문제를 겪었다(510 선언 / 507 디코드).

## 2. 스코프

**포함**
- `detector/psecore_adapter.py` (신규) — psecore를 기존 detector 인터페이스로 번역
- 소비자(`prior/`, `training/`, `mitigator/`, `scripts/`)가 어댑터를 쓰도록 전환
- 기존 검출기와의 판정 차이 실측, 특히 합성 파이프라인 수용률

**제외 (별도 작업)**
- 재학습 — 검출기 교체 후 별도로 진행
- RGB 채널 축 활성화 (청색 점멸 탐지) — psecore 기본값 `warn` 유지. 도입 후 별도 판단
- 마스크 지속 로직 — 이미 구현 완료(커밋 `0c2cf8f`), 이 작업과 독립적으로 그대로 얹힌다
- `external/psecore.py` 수정 — 팀원 코드이므로 우리는 건드리지 않는다

## 3. 사전 측정 결과 (이 설계의 근거)

2026-08-05에 실측한 값들이며, 설계 결정의 근거다.

**psecore의 `over` 마스크는 보정용으로 부적합하다.** `FlashCounter`가 화소별로 유지하는 `over = cnt > max_flashes_per_sec`가 "지속되는 위험 영역" 마스크로 쓸 만해 보였으나, 실측 결과 오히려 나빴다:

| 마스크 | 손상 화소 커버율 (평균/중앙값) | 마스크 면적 |
|---|---|---|
| 현재 전환 마스크 | 35.9% / 14.5% | 18.1% |
| 우리 0.2초 OR (구현 완료) | **87.6% / 93.2%** | 46.9% |
| psecore `over` | 23.3% / 20.1% | 14.7% |

`over`는 "허용치보다 **빠르게** 번쩍이는 화소"라, 허용치 이하 속도로 번쩍이면서도 화면이 망가진 화소를 놓친다. **`over`는 위험 판정용 지표이지 보정 대상 마스크가 아니다.** 따라서 psecore에 `over` 노출을 요청할 필요가 없고, 마스크 지속은 우리 0.2초 OR을 유지한다.

**마스크 지속 로직은 검출기와 독립적이다.** OR은 전환 마스크 위에 얹히므로, 전환 마스크를 psecore 것으로 갈아끼워도 그대로 작동한다.

## 4. 구조

```
external/psecore.py            (팀 코드, 수정하지 않음)
        │ import
        ▼
detector/psecore_adapter.py    (신규 — 번역만 담당)
        │ 기존과 동일한 시그니처 제공
        ▼
prior/ · training/ · mitigator/ · scripts/   (import 경로만 변경)
```

어댑터는 기존과 **완전히 같은 시그니처** 두 개를 제공한다:

```python
def run_detection(
    frames: Iterable[np.ndarray], fps: float, profile: ThresholdProfile,
    margin_seconds: float = 0.5,
) -> tuple[list[FlickerScore], list[RiskSegment]]

def run_detection_with_masks(
    frames: Iterable[np.ndarray], fps: float, profile: ThresholdProfile,
    margin_seconds: float = 0.5,
) -> tuple[list[FlickerScore], list[RiskSegment], list[np.ndarray]]
```

`detector/pipeline.py`는 **삭제하지 않는다.** 어느 구현을 쓸지 선택 가능하게 두어, 문제 발생 시 즉시 되돌리고 두 결과를 나란히 비교할 수 있게 한다.

## 5. 데이터 변환

### 5.1 프레임 형식

우리: RGB float32 `[0,1]`. psecore: BGR uint8 (`_LIN[bgr_u8]` LUT 인덱싱).

```python
bgr_u8 = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
```

uint8 변환에 따른 정밀도 손실은 감수한다. 합성 재검증에 영향이 있는지는 §7-③에서 확인한다.

psecore의 내부 다운스케일(`Cfg.short_side`, 기본 240)은 **어댑터 경로에서는 발생하지 않는다.** 리사이즈는 `open_video`/`iter_frames`에서만 일어나는데, 우리는 프레임을 직접 넘기므로 그 경로를 타지 않는다. 마스크는 원본 해상도로 나온다.

**단, §7-① 동등성 테스트에서는 이것이 함정이 된다.** 그 테스트는 `analyze(파일)`과 어댑터(프레임)를 비교하는데, `analyze` 쪽만 240으로 다운스케일되면 애초에 다른 해상도를 비교하게 된다. 테스트에서는 `short_side`를 프레임의 짧은 변 이상으로 올려 리사이즈 분기를 타지 않게 하거나, 처음부터 짧은 변이 240 이하인 테스트 영상을 쓴다.

### 5.2 프로파일 → Cfg

| ThresholdProfile | psecore Cfg | 역할 |
|---|---|---|
| `max_flashes_per_second` | `max_flashes_per_sec` | 초당 허용 플래시 수 |
| `max_area_ratio` | `area_ratio` | 허용 면적 비율 |
| `general_flash_delta_threshold` | `theta_lum` | 전환 크기 임계 |
| `general_flash_dark_threshold` | `theta_dark_max` | 어두운쪽 상한 |
| `red_saturation_ratio_threshold` | `red_ratio` | 적색 채도 임계 |

나머지 `Cfg` 필드는 psecore 기본 프로파일(`PROFILES["bt1702"]`) 값을 그대로 쓴다.

**주의**: 이름을 맞춘 게 아니라 역할이 대응하는 매핑이다. 계수 방식 자체가 다르므로(peak-valley 쌍 vs 프레임 카운팅) **판정 결과는 달라진다.** 그것이 도입 목적이다.

### 5.3 채널 → 단일 마스크

psecore는 6채널(LUM/RGB/RED/RG/BY/RB) 마스크를 따로 낸다. 어댑터는 **해당 프레임에서 FAIL 판정에 기여한 채널만 OR**한다 — "위험하다고 판정한 근거를 고친다"로 자기일관적이고, `Cfg`의 `chroma_mode` 등으로 채널을 끄면 마스크에서도 자동으로 빠진다.

psecore 자신이 명시한 채널별 신뢰도(참고):

| 채널 | 기본 모드 | psecore 자체 평가 |
|---|---|---|
| LUM, RED | fail | 표준 원문 확정 |
| RB | fail | 임상 근거 있음 (Parra 2007) |
| RG, BY | fail | **임계값은 임상 근거 없는 추정치** |
| RGB | **warn** | 표준에 없는 신규 축, FAIL 판정 미반영 |

기본값에서 RGB는 warn이라 마스크에서 제외된다.

### 5.4 결과 → 우리 타입

- psecore `Segment`는 밀리초 단위 → `RiskSegment(start_frame, end_frame)`로 변환: `round(ms / 1000 * fps)`
- `margin_seconds`는 psecore가 다루지 않으므로 어댑터가 변환 후 구간을 확장한다 (기존 `scores_to_segments`와 동일하게 `margin_frames = round(margin_seconds * fps)`, 클립 경계로 클램프, 겹치면 병합)
- `FlickerScore`: `detector/` 밖에서 실제로 읽는 필드는 **`frame_index` 하나뿐**이다 (전수 확인: `prior/compute.py:131`이 유일한 사용처). `uncertain`은 target histogram이 시간축에서 공간축으로 바뀌면서(커밋 `7e74e17`) 소비처가 사라졌다. 나머지 필드(`flagged_frame_count_last_second`, `flagged_area_ratio`, `max_flagged_area_in_window`, `uncertain`)는 psecore에서 얻을 수 있는 대응값으로 채우되, 대응값이 없으면 0/False로 둔다. **채우지 못한 필드는 어댑터 docstring에 명시**해, 나중에 누군가 그 값을 신뢰하는 일이 없게 한다.

### 5.5 프레임 입력 문제

`psecore.analyze(path, ...)`는 **파일 경로만 받는다.** 그러나 우리는 메모리 위의 프레임을 검사해야 하는 곳이 있다:

- `training/synth.py` — 방금 주입한 손상 프레임을 즉시 재검증 (파일이 없다)
- `prior/compute.py` — 이미 디코딩된 프레임

임시 파일 우회는 채택하지 않는다. 합성 재검증은 샘플당 최대 5회 반복(172샘플 × 12조합)이라 인코딩/디코딩 비용이 크고, **인코딩 손실로 픽셀값이 바뀌어 "주입한 그대로를 검사한다"는 의미가 깨진다.**

**우선안**: 팀원에게 순수 리팩터 요청 — 동작 변경 없이 프레임 루프를 분리한다.

```python
def analyze_frames(frames, fps, cfg=None, want_masks=False):
    ...  # 기존 루프 본문을 그대로 이동

def analyze(path, cfg=None, want_masks=False):   # 기존 호출자 영향 없음
    cap, info = open_video(path, cfg.short_side)
    return analyze_frames(iter_frames(cap, info), info.fps, cfg, want_masks)
```

**대안** (요청이 불가능한 경우): 어댑터가 psecore의 구성요소(`PeakValley`, `BinaryTransition`, `FlashCounter`, `sync_groups`, `area_global`, `area_wcag`, `luminance`, `linear_rgb`, `uv_prime`, 변환 행렬·상수)를 import해 루프만 재현한다. 알고리즘은 전부 psecore 것이고 우리 것은 I/O와 루프 뼈대뿐이다. 이 경우 §7-①의 동등성 테스트가 **필수**가 된다.

## 6. 에러 처리

- psecore가 던지는 예외(`IOError` 등)는 어댑터가 잡아서 삼키지 않고 그대로 전파한다 — 검출 실패를 "안전함"으로 보고하는 일은 절대 없어야 한다는 기존 원칙과 같다.
- 프로파일 매핑에서 대응 필드가 없거나 값이 범위를 벗어나면 즉시 `ValueError`를 던진다. 조용히 기본값으로 대체하지 않는다.
- `training/cli.py`의 배치 루프는 이미 `ValueError`를 샘플 단위로 건너뛰도록 처리되어 있다(커밋 `0ed2253`).

## 7. 검증 전략

"판정 결과 일치"가 목적이므로, 결과가 어떻게 달라지는지 측정하지 않으면 도입 의미가 없다.

**① 동등성 테스트** (§5.5 대안 채택 시 필수)
`psecore.analyze(파일)`과 어댑터(디코딩된 프레임)가 같은 영상에서 동일한 위험 구간을 내는지 검증한다. psecore가 나중에 변경되면 이 테스트가 깨져 알려준다.

**② 기존 검출기와의 차이 실측**
같은 영상 집합을 양쪽에 통과시켜 위험 구간의 개수·길이 차이를 기록한다. 2026-08-04 테스트 클립(`test_mp4/`)과 `data/clips`의 소스 클립을 쓴다.

**③ 합성 파이프라인 수용률 회귀 확인** (가장 중요)
`training/synth.py`는 주입 결과를 Detector로 재검증해 통과한 것만 채택한다. 검출기가 바뀌면 수용률이 무너질 수 있고, 그러면 학습 데이터가 사라진다. 현재 기준선은 **87개 클립 → 172샘플**이다. 교체 후 이 수치를 재측정해, 크게 떨어지면 프로파일 매핑(§5.2)을 조정한다.

**④ 어댑터 자체 단위 테스트**
- 프레임 형식 변환이 값을 보존하는지 (RGB float → BGR uint8 왕복)
- 프로파일 매핑이 각 필드를 올바른 `Cfg` 필드로 옮기는지
- 밀리초 → 프레임 변환과 margin 확장이 기존 `scores_to_segments`와 같은 규칙을 따르는지
- 채널 OR이 warn 채널을 제외하는지

**⑤ 기존 테스트 230개**
`detector/pipeline.py`를 남기므로 기존 detector 테스트는 계속 통과해야 한다. 어댑터는 자체 테스트를 새로 갖는다.

## 8. 리스크

| 리스크 | 대응 |
|---|---|
| **합성 수용률 급감** — 학습 데이터가 사라짐 | §7-③에서 먼저 측정. 심하면 §5.2 매핑 조정 |
| psecore 판정이 더 엄격/느슨해 기존 학습 데이터가 무효화 | 기존 detector를 남겨두므로 즉시 되돌리기 가능 |
| uint8 변환 정밀도 손실이 합성 재검증에 영향 | §7-③에서 확인 |
| psecore 내부 다운스케일로 마스크 해상도 불일치 | §5.1대로 `short_side`를 끈다 |
| 팀원 리팩터를 못 받아 루프를 재현하게 됨 | §7-① 동등성 테스트로 어긋남 감시 |

## 9. 성공 기준

1. 어댑터가 기존 시그니처를 그대로 만족하고, 소비자 코드는 import 경로만 바뀐다
2. §7-③ 합성 수용률이 기존(87클립 → 172샘플) 대비 크게 떨어지지 않는다
3. §7-② 기존 검출기와의 판정 차이가 측정되어 문서화된다
4. 기존 테스트 230개가 계속 통과한다

**이 작업의 성공 기준에 "보정 성능 향상"은 포함되지 않는다.** 검출 기준 통일이 목적이며, 보정 성능은 재학습 이후 별도로 평가한다.
