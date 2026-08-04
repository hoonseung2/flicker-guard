# 위험 구간 배치 완화 + 마스크 모양/움직임 다양화 — 설계 문서

## 1. 배경과 목적

Mitigator를 실제로 H100에서 학습시키기 전에, 최종 브랜치 리뷰에서 실측으로 발견된 두 가지 문제를 DatasetSynth 쪽에서 고친다.

**문제 1 — 학습 예시 수율**: 실제 170개 샘플 중 153개(90%)가 학습 예시를 0개 만들고, 실질적으로 15개 클립에서만 예시가 나온다(`docs/superpowers/specs/2026-08-03-mitigator-design.md` §8에 이미 기록됨). 원인을 실측으로 추적한 결과: 예시를 만든 샘플들은 위험 구간이 클립의 42~59% 지점에서 시작했고(중앙값 39프레임), 예시를 못 만든 샘플들은 15% 지점(중앙값 10프레임)에서 시작했다. `sample_synthesis_params`가 위험 구간 시작 위치를 클립 전체 범위에서 균등 분포로 뽑다 보니, PriorCalc의 `TargetHistogramSmoother`가 목표 히스토그램을 계산할 만큼 충분한 "클린 프레임"을 위험 구간 앞에 확보하지 못하는 경우가 잦았다 — 클립 길이 자체의 문제가 아니라 **배치 위치**의 문제다.

**문제 2 — 마스크 다양성**: 지금 위험 구간의 공간 마스크는 프레임 중앙의 정적 사각형 하나뿐이다(`training/params.py`의 `_mask_dims`). `team-overview.md`가 AI가 필요한 이유로 든 "실제 위험 패턴은 깔끔한 도형이 아니라 장면과 함께 움직이는 불규칙한 광원"이라는 전제와 학습 데이터가 어긋나 있다 — Mitigator가 "정적 사각형만 잘 복원하는" 편향을 학습할 위험이 있다. 마침 데모 영상으로 쓸 콘서트 조명 영상도 이 형태(움직이는 스포트라이트/빔, 여러 개 동시)와 맞아떨어진다.

두 문제 모두 실제 학습을 시작하기 전에 고치기로 결정했다(`docs/lessons-learned.md`에도 두 사례로 기록됨).

## 2. 스코프

**포함**:
- `training/params.py`의 `sample_synthesis_params`: 위험 구간 시작 위치를 "클린 러너웨이 확보 가능하면 확보, 안 되면 가능한 만큼만" 규칙으로 재샘플링
- `training/params.py`: 조명 1~N개, 각각 모양(사각형/원/빔)·크기·시작 위치·끝 위치(직선 스윕)를 샘플링하는 로직 추가. `InjectionWindow`가 이 조명 목록을 담도록 확장
- `training/injection.py`의 `inject_general_flash_realistic`/`inject_red_flash_realistic` 두 함수만: 정적 사각형 슬라이스 대신, 프레임별로 각 조명의 현재 위치를 보간해 모양을 렌더링하고 합집합한 마스크에 배율 적용
- `docs/superpowers/specs/2026-08-03-mitigator-design.md` §8: 수율 개선 결과를 실측 후 갱신

**제외 (범위 밖)**:
- `inject_general_flash`/`inject_red_flash`(flat 모드) — 안 건드림. flat은 단순 폴백/레거시 용도로 유지
- `general`/`red` 외 새로운 색상 카테고리(파랑·보라 등 임의 색 조명) — 이번 스코프 아님. 색상은 지금처럼 general(무채색 밝기, realistic 배율로 질감 보존)과 red(WCAG 채도 규정 전용)만 유지
- 조명 간 비동기 펄스(각자 다른 타이밍으로 깜빡임) — v1은 한 샘플 안의 모든 조명이 `period_frames`를 공유하고 동기화되어 펄스. 실제 콘서트 조명도 비트에 맞춰 동기화되는 경우가 흔해 충분한 근사로 판단
- 겹침을 고려한 정확한 픽셀 단위 면적 계산 — 여러 조명의 면적은 "겹치지 않는다고 가정한 합"으로 보수적으로 추정. 정확한 겹침 처리는 필요성이 실측되면 추후 검토
- 클립 추가 소싱(DAVIS 외 새 영상) — 별도 작업으로, 이번 결과를 보고 필요성 판단

## 3. 배치 위치 완화 로직

```python
latest_start = clip_frame_count - duration_frames   # 기존 로직, 변경 없음
if latest_start < 1:
    raise ClipTooShortError(...)                     # 기존 기준, 변경 없음

desired_runway = window_frames + _RUNWAY_MARGIN       # PriorCalc가 필요로 하는 클린 프레임 수
min_start = min(desired_runway, latest_start)         # 못 채우면 채울 수 있는 만큼만
start_frame = int(rng.integers(min_start, latest_start + 1))
```

`desired_runway`는 Detector의 `WindowedFlashCounter` 워밍업 길이(`window_frames = round(fps)`)에 마진을 더한 값으로, 정확한 마진 크기는 구현 단계에서 실제 클립으로 검증해 확정한다(실측 근거: 기여한 샘플들의 시작 위치 중앙값이 `window_frames`의 약 1.5~2배였음).

클립이 충분히 길면(`latest_start >= desired_runway`) 완전한 러너웨이를 확보한 채로 기존처럼 뒤쪽 구간 안에서 무작위성이 유지된다. 클립이 짧으면 `start_frame`이 `latest_start`(가능한 가장 늦은 위치)로 강제되어, 완전하진 않아도 최대한의 러너웨이를 확보한다 — `ClipTooShortError`가 발생하는 최소 클립 길이 기준 자체는 바뀌지 않는다.

## 4. 마스크 모양/움직임

**조명 표현**: 각 조명은 모양 종류(사각형/원/빔) + 크기 + 시작 위치 + 끝 위치를 가진다. 위험 구간의 `start_frame`부터 `end_frame`까지 각 조명의 중심 위치가 시작→끝을 직선 보간하며 이동한다(크기는 스윕 동안 고정 — 위치만 움직임). 한 샘플에 조명이 여러 개(N, 랜덤) 있을 수 있고, 각 조명은 독립적인 경로를 가지되 전부 같은 `period_frames`로 동기화되어 펄스한다.

**저장 방식**: `meta.json`에는 각 조명의 파라미터(모양·크기·시작/끝 위치)만 저장한다 — 프레임별 픽셀 마스크를 통째로 저장하지 않는다. 실제 프레임에 주입할 때(`inject_general_flash_realistic`/`inject_red_flash_realistic`) 그 프레임 번호에 맞춰 각 조명의 현재 위치를 계산하고, 그 프레임만의 마스크를 그때그때 렌더링해서 합집합한다.

이 방식을 택한 이유: (a) `meta.json`이 지금처럼 작고 사람이 읽기 쉽게 유지됨, (b) 디스크 사용량이 늘지 않음(현재 데이터셋만으로도 15GB), (c) 어차피 Detector/PriorCalc는 실제 픽셀 변화에서 자기 마스크를 독립적으로 재계산하므로 — DatasetSynth가 만든 "정답 마스크"를 프레임 단위로 저장해서 넘겨줄 필요가 애초에 없다(`MitigatorDataset`은 `injected_window`의 좌표가 아니라 `prior.compute.compute_prior`가 계산한 마스크만 사용).

**면적비 검증**: 조명이 여러 개일 때 목표 면적(`profile.max_area_ratio` 마진 포함)은 "겹치지 않는다고 가정한 각 조명 면적의 합"으로 보수적으로 계산한다. 실제 렌더링 시 조명들이 겹치면 실제 노출 면적은 이보다 작을 수 있지만, 최종 검증은 지금과 동일하게 `synthesize_sample_realistic`이 실제로 `Detector.run_detection`을 돌려 확인하고, 기준 미달이면 재샘플링·재시도(최대 5회, 기존 메커니즘 그대로)한다.

## 5. 데이터 흐름

```
sample_synthesis_params(profile, pattern, clip_frame_count, fps, rng)
  → 위험 구간 [start_frame, end_frame] (완화된 배치 로직)
  → 조명 목록 [(모양, 크기, 시작위치, 끝위치), ...] (N개, 랜덤)
  → InjectionWindow(start_frame, end_frame, lights=[...], period_frames, ramp_frames)

inject_general_flash_realistic(frames, window, gain_dark, gain_bright)
inject_red_flash_realistic(frames, window, red_gains, baseline_gains)
  → 각 프레임 i에 대해:
      각 조명의 현재 위치 = 시작위치 + (끝위치-시작위치) * (i-start_frame)/(end_frame-start_frame)
      각 조명의 모양을 그 위치에 렌더링 → 불린 마스크
      전체 마스크 = 조명 마스크들의 합집합
      전체 마스크 영역에 배율 적용 (기존 realistic 로직 그대로: 선형 공간 곱셈)

synthesize_sample_realistic(...)  # 변경 없음: Detector로 재검증, 실패 시 재시도
write_sample(...)  # meta.json에 조명 파라미터 저장 (스키마 확장)
```

## 6. 에러 처리

- `ClipTooShortError` 발생 기준(`clip_frame_count < duration_frames + 1`)은 변경 없음 — 배치 완화는 이 최소 기준을 넘은 클립 안에서 위치를 어떻게 고르는지만 바꾼다.
- 조명 구성(모양·크기·경로)이 5번 재시도 안에 Detector 임계값을 못 넘기면 기존과 동일하게 `validation_exhausted`로 실패 처리 — 새 실패 유형을 추가하지 않는다.
- 빔/원이 프레임 경계를 벗어나는 위치로 보간되면 렌더링 시 프레임 범위로 클램핑한다(화면 밖 부분은 그리지 않음).

## 7. 테스트 전략

Detector·DatasetSynth·PriorCalc·Mitigator와 동일한 스타일: 작은 인메모리 합성 데이터로 CI에서 결정론적으로 검증. 실제 DAVIS 데이터는 CI 대상이 아니며 구현 완료 후 수동으로 실행.

- 모양별 렌더링 함수: 사각형/원/빔 각각 기대한 픽셀 면적을 만드는지, 프레임 경계 클램핑이 올바른지
- 직선 보간: `start_frame`에서 정확히 시작 위치, `end_frame`에서 정확히 끝 위치, 중간 프레임에서 선형적으로 이동하는지
- 배치 완화 로직: 긴 클립(러너웨이 확보 가능)과 짧은 클립(러너웨이 부족, `latest_start`로 강제) 양쪽 경계 케이스
- 여러 조명의 마스크 합집합이 올바르게 계산되는지(개별 조명 마스크의 논리합과 일치)
- 작은 합성 클립으로 `synthesize_sample_realistic`이 새 조명 구성(단일/복수, 모양별)에서도 검증을 통과하는지(기존 재시도 루프가 여전히 작동하는지)
- **구현 완료 후 수동 검증** (자동 테스트 아님): 실제 90개 클립 전체 재합성 → 수율(기여 샘플 수·클립 수) 실측, `2026-08-03-mitigator-design.md` §8 갱신

## 8. 열린 리스크 / 향후 로드맵

- **`desired_runway` 마진값은 잠정치**: 실제 클립으로 검증 후 조정 예정 (§3)
- **조명 간 겹침 미고려**: 면적비를 보수적으로(비겹침 가정) 계산하므로 실제로는 목표보다 더 넉넉하게 위험할 수 있음 — 문제라고 판단되면 추후 겹침 회피 제약 추가 검토
- **비동기 펄스**: v1 제외, 실제 학습 결과에서 필요성이 보이면 재검토
- **클립 소싱 확장**: 이번 개선(배치+모양 다양화)만으로 90개 클립 규모에서 첫 학습을 돌려보고, 그 결과를 보고 새 클립 소싱 여부 판단
