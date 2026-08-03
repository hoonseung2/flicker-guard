# PriorCalc — Design Spec

## 1. 배경과 목적

README 아키텍처(`Detector → PriorCalc(조명 prior) → Mitigator`)의 두 번째 단계. Mitigator(Tier1 완화망, Flickerformer 백본)가 위험 구간을 보정할 때 참고할 "조명 prior"를 계산한다. 아이디어 출처는 BlazeBVD(arXiv:2403.06243, 코드 비공개) — STE(Scale-Time Equalization, 고전적 히스토그램 기반 디플리커링 기법)를 이용해 프레임별 조도 히스토그램을 시간축으로 스무딩, 신경망의 학습 부담을 줄이는 접근.

**핵심 설계 원칙**: BlazeBVD의 원래 3-출력(전역 히스토그램, singular frames set, 국소 exposure map) 중 뒤의 둘은 새로 만들지 않고 **Detector가 이미 계산해둔 신호를 재사용**한다.
- "어느 프레임이 위험한가"(singular frames set에 해당) → Detector의 `FlickerScore`/`RiskSegment` 재사용
- "어느 픽셀이 위험한가"(exposure map에 해당) → Detector가 프레임마다 계산만 하고 버리던 픽셀 단위 `transition_mask`/`red_flash_mask` 결과를 노출시켜 재사용

이렇게 하는 이유: BlazeBVD의 exposure map은 "절대적 노출 품질"(정적 프레임에서도 성립)을 다루는 범용 신호인 반면, 우리가 실제로 필요한 건 "프레임 간 변화가 PSE 위험 기준을 넘었는가"라는 시간적 transition 신호다. Detector가 정확히 이걸 계산하고, 이미 검증까지 끝났다. 새로 설계하면 Detector 개발 때 겪었던 임계값·경계 케이스 검증을 처음부터 반복하게 된다.

PriorCalc가 새로 계산하는 것은 딱 하나 — **"위험하지 않았다면 조도가 어떤 궤적이었을지"를 나타내는 시간축 스무딩된 타겟 히스토그램**이다. 이는 Detector가 전혀 제공하지 않는 정보다(Detector는 위험 여부만 yes/no로 판단하지, "정상이면 이래야 한다"는 알려주지 않는다).

## 2. 스코프

**포함**:
- 프레임별 조도 히스토그램 계산 (`detector.luminance.relative_luminance` 재사용)
- 시간축 스무딩: 트레일링 윈도우(Detector와 동일하게 `round(fps)`, 약 1초) 안에서 이동평균을 낼 때 Detector가 위험/불확실로 판정한 프레임은 제외
- `detector.pipeline.run_detection`의 작은 인터페이스 확장 — 지금 계산 후 버리는 프레임별 픽셀 마스크를 결과에 포함 (기존 반환값·시그니처는 하위 호환 유지)
- 두 가지를 합친 `compute_prior` 오케스트레이터: 프레임별 (스무딩된 타겟 히스토그램, 위험 픽셀 마스크) 쌍을 출력

**제외 (향후 로드맵으로만 명시)**:
- BlazeBVD식 독립 exposure map 알고리즘 — Detector 마스크 재사용으로 대체하기로 결정, 새로 설계하지 않음
- Mitigator 자체 — 별도 설계/계획
- **WCAG "10도 시야각 서브셋" 면적 기준 미반영 문제** (README §9에 새로 기록됨): Detector의 `flagged_area_ratio`가 화면 전체 대비로 계산되는데, 실제 WCAG 기준은 10도 시야각 서브셋 내 25%임. 이 스펙에서 고치지 않음 — PriorCalc/Mitigator 구현 착수 직전에 반영 여부를 별도로 확인하기로 함(진행 상황 tracking에 이미 기록됨).

## 3. 모듈 위치

```
flicker-guard/
├── prior/
│   ├── __init__.py
│   ├── histogram.py        # 조도 히스토그램 계산 + 시간축 스무딩
│   └── tests/
│       └── test_histogram.py
└── detector/
    └── pipeline.py          # run_detection 확장 (기존 파일 수정)
```

## 4. 컴포넌트

| 함수/클래스 | 역할 |
|---|---|
| `compute_illumination_histogram(luminance, n_bins=64)` | 프레임 하나의 상대 조도 배열 `(H, W)` → 정규화된 히스토그램 `(n_bins,)`. Detector의 `relative_luminance` 출력을 그대로 입력받음 |
| `TargetHistogramSmoother(window_frames)` | Detector의 `WindowedFlashCounter`와 같은 스타일의 스테이트풀 클래스. `update(frame_index, histogram, is_risky_or_uncertain) -> np.ndarray \| None`: 트레일링 윈도우 안의 "위험하지 않은" 프레임들의 히스토그램만으로 이동평균을 내 스무딩된 타겟을 반환. 윈도우 안에 clean 프레임이 없으면 가장 최근 clean 프레임의 히스토그램으로 폴백. 지금까지 clean 프레임을 하나도 못 봤으면 `None` 반환 |
| `compute_prior(frames, fps, profile)` | 오케스트레이터. 확장된 `detector.pipeline.run_detection`을 호출해 프레임별 위험 판정 + 픽셀 마스크를 받고, 위 둘을 엮어 프레임별 `(target_histogram, pixel_mask)` 쌍의 리스트를 반환 |
| `detector.pipeline.run_detection` (확장) | 기존 반환값(`list[FlickerScore]`, `list[RiskSegment]`)에 프레임별 픽셀 마스크(`list[np.ndarray]`, 각 `(H, W)` bool)를 추가로 반환하도록 확장. 기존 필드는 전혀 안 바뀜 |

## 5. 데이터 흐름

```
frames (clip)
  → detector.pipeline.run_detection (확장판)
       → 프레임별: FlickerScore + RiskSegment 목록 (기존과 동일)
       → 추가: 프레임별 pixel mask (H, W) bool — 어느 픽셀이 flicker를 유발했는지
  → prior.compute_prior
       → 각 프레임의 relative_luminance → compute_illumination_histogram → 히스토그램
       → TargetHistogramSmoother.update(frame_index, histogram, is_risky)
            — is_risky는 다음 둘 중 하나라도 참이면 True: (a) 해당 프레임 인덱스가 반환된 RiskSegment 중 하나의 [start_frame, end_frame] 범위 안에 속함, (b) 해당 프레임의 FlickerScore.uncertain이 True. 즉 "위험 구간에 속함" 또는 "판정 근거가 아직 불확실함" 중 하나만 만족해도 스무딩 대상에서 제외
       → 프레임별 (스무딩된 타겟 히스토그램, 위험 픽셀 마스크) 쌍
  → 출력: Mitigator가 conditioning으로 쓸 재료
```

## 6. 에러 처리

- **윈도우 안에 clean 프레임 없음** (짧은 지속 스트로브 등): 가장 최근 clean 프레임의 히스토그램으로 폴백
- **클립 전체에 clean 프레임이 하나도 없음**: 폴백할 곳이 없으므로 해당 프레임(들)은 타겟 `None`으로 표시 — 조용히 이상한 값을 만들지 않음. 이런 클립은 Mitigator 학습에서 별도 처리(제외 등) 필요함을 의미
- **프레임 0 / 경계 프레임**: Detector의 I4 수정으로 프레임 0은 항상 `uncertain=True`이므로 자동으로 스무딩 대상에서 제외됨 (별도 특별 처리 불필요, 기존 규칙이 자연 커버)
- **Detector 인터페이스 확장의 하위 호환성**: 기존 `FlickerScore`/`RiskSegment` 필드·시그니처는 절대 변경 안 함. 기존 Detector 66개 테스트와 DatasetSynth `training/` 코드가 전혀 안 깨져야 함 — 구현 계획에서 명시적으로 검증

## 7. 테스트 전략

Detector·DatasetSynth와 동일한 스타일: 작은 인메모리 합성 프레임만으로 CI에서 결정론적으로 검증.

- `compute_illumination_histogram`: 알려진 밝기 분포(예: 절반 밝음/절반 어두움)를 넣고 예상 bin이 채워지는지 확인
- `TargetHistogramSmoother`: 
  - 인위적으로 "이 프레임만 확 밝은" 시퀀스에서, 그 프레임을 위험으로 표시했을 때 스무딩된 타겟이 그쪽으로 안 끌려가고 주변 clean 프레임 평균과 같은지 검증 (핵심 안전장치)
  - 윈도우 안에 clean 프레임이 없을 때 가장 최근 clean 프레임으로 폴백하는지 검증
  - 클립 전체에 clean 프레임이 없을 때 크래시 없이 `None`을 반환하는지 검증
- `compute_prior` 통합테스트: DatasetSynth의 `inject_general_flash`/`inject_red_flash`(이미 구현됨)로 "이 구간에 정확히 이 위험을 주입했다"는 걸 아는 합성 클립을 만들고, 반환된 픽셀 마스크가 주입 영역과 일치하는지, 스무딩된 타겟이 주입 전 원래 밝기를 대표하는지 확인
- `run_detection` 확장 회귀 테스트: 기존 66개 Detector 테스트가 그대로 통과하는지(하위 호환) + 새로 노출된 픽셀 마스크가 `transition_mask`/`red_flash_mask`를 독립적으로 계산한 것과 일치하는지 검증

## 8. 열린 리스크 / 향후 로드맵

- **WCAG 10도 시야각 서브셋 면적 기준**: README §9에 기록됨. PriorCalc/Mitigator 구현 착수 직전 재확인 필요 (진행 상황 tracking에 체크포인트로 등록됨).
- **폴백 히스토그램 품질**: "가장 최근 clean 프레임" 폴백은 그 프레임이 시간적으로 멀리 떨어져 있으면(예: 클립 앞부분에 긴 위험 구간) 부정확할 수 있음. 실제 DAVIS 데이터로 PriorCalc를 돌려본 뒤 폴백 빈도·품질을 실측해서 필요시 개선.
- **히스토그램 bin 개수·정규화 방식**: 이번 설계는 구체적 수치(예: bin 개수)를 구현 계획 단계에서 확정. Mitigator 설계가 나온 뒤 conditioning 입력으로 적합한 차원인지 재검토 필요할 수 있음.
