# DatasetSynth 합성 파라미터 다양화 — 설계 문서

## 1. 배경

2026-08-04 실제 콘서트 조명 영상(`test_mp4/`, 약 17초)으로 학습된 Mitigator를 검증한 결과, 보정 강도가 안전 기준에 크게 못 미쳤다. `prior/compute.py`의 target histogram 알고리즘 결함을 고치고(커밋 `7e74e17`, `b2e7d04`) 재학습했지만:

- 프레임간 휘도 급변 감소율: 9.7% (재학습 전후 변화 없음)
- 휘도 시간축 변동폭 감소율: 5.8% → 10.9% (개선은 있음)
- **보정 결과를 `run_detection`에 다시 통과시켰을 때 위험 구간이 여전히 2개** (3.4~16.9초 → 4.0~16.9초로 0.6초만 축소) — 즉 재검증 기준으로는 사실상 실패

따라서 병목은 알고리즘이 아니라 **학습 데이터의 분포가 실제 영상에 비해 지나치게 좁다는 것**으로 좁혀졌다. 실측 비교:

| 항목 | 현재 학습 데이터 | 실제 테스트 영상 |
|---|---|---|
| 마스크 면적 | 프로파일 임계값 바로 위 고정 (~0.10~0.25) | 최대 74.8% |
| general 밝기 대비 | 고정 공식 1개 값 | — |
| red 채널 R값 | `r = 0.9` 하드코딩 | 0.17~0.50 사이 변동 |
| red 베이스라인 | `(0.3, 0.3, 0.3)` 하드코딩 | 중성 회색 아님 |

## 2. 스코프

**포함** (전부 `training/params.py`의 샘플링 로직 수정으로 가능):
- 마스크 면적을 `[프로파일 임계값 + 마진, 0.60]` 구간에서 균일 샘플링
- general 패턴의 `dark_target` / `bright_target` 랜덤 샘플링
- red 패턴의 `r`, `baseline_rgb`, `target_ratio` 랜덤 샘플링
- 위 변경으로 노출되는 기존 `sample_lights` ValueError 문제 해결

**제외 (긴 소스 영상 확보가 선행되어야 함)**:
- 위험 구간 지속시간 다양화 (8~15초 연속) — 현재 소스 클립이 최대 104프레임(24fps, 4.3초)이라 물리적으로 불가능
- 한 클립 내 다중 위험 구간 — 한 구간에 약 84프레임(러닝웨이 48 + 윈도우 36)이 필요해 두 구간은 120프레임 이상 요구, 현재 클립 길이 초과
- 소스 클립 도메인 교체 (콘서트/무대 영상)

위 세 가지는 모두 "더 긴 소스 영상 확보"라는 동일한 선행 조건에 묶여 있으므로 별도 후속 프로젝트로 다룬다.

**제외 (별도 판단)**:
- MitigatorNet 아키텍처/파라미터 수 변경 — 데이터 다양화 후 재평가

## 3. 마스크 면적 다양화

### 3.1 현재 동작

`sample_lights` (training/params.py)는 목표 총 면적을 고정 계산한다:

```python
target_total_area = min(
    (profile.max_area_ratio + _AREA_MARGIN) * frame_height * frame_width,
    _MAX_LIGHTS_AREA_RATIO * frame_height * frame_width,
)
```

프로파일이 정해지면 항상 같은 값이므로, 모든 학습 예제의 마스크 면적이 사실상 동일하다.

### 3.2 변경

목표 면적 비율을 `[profile.max_area_ratio + _AREA_MARGIN, _MAX_LIGHTS_AREA_RATIO]` 구간에서 균일 샘플링한다. 균일 분포를 쓰는 이유는 작은 국소 반짝임부터 화면 대부분을 덮는 경우까지 같은 빈도로 학습시키기 위함이다.

### 3.3 필수 동반 수정: beam 실패율 문제

`training/params.py`에 이미 실측 기록된 문제가 있다 — 목표 면적이 커지면 `sample_lights`가 일반 `ValueError`를 던지는 비율이 급증한다(측정: ≤0.40 → 0%, 0.45~0.50 → 약 11.5%, 0.55 → 약 19.5%, 0.60 → 약 33%).

원인은 `_sample_one_light`의 clamp다. 모양별 크기가 프레임 절반(`max_half`)으로 제한되어 큰 목표 면적을 채우지 못하고, 그 결과 `_require_exceeds`의 합산 면적 검증이 실패한다. 이 `ValueError`는 `ClipTooShortError`가 아니므로 `training/cli.py`의 배치 러너가 잡지 않아 **배치 실행 전체가 중단된다**.

면적 샘플링 범위를 0.60까지 넓히면 이 실패 경로를 정면으로 밟게 되므로 함께 고쳐야 한다.

480x854 프레임에서 실측한 모양별 달성 가능 면적 비율:

| 라이트 1개당 목표 | beam | rect | circle |
|---|---|---|---|
| 0.40 | 0.401 ✓ | 0.400 ✓ | 0.398 ✓ |
| 0.50 | 0.473 ✗ | 0.501 ✓ | 0.441 ✗ |
| 0.60 | 0.519 ✗ | 0.564 ✓ | 0.441 ✗ |

beam뿐 아니라 **circle도 반지름 clamp 때문에 약 0.441에서 상한에 걸린다**. 큰 단일 목표 면적을 달성할 수 있는 것은 rect뿐이다. (라이트를 여러 개 뽑는 경우 목표가 개수만큼 나뉘므로 이 상황은 라이트 1~2개일 때 주로 발생한다.)

**해결책 (2단계)**:

1. **근본 원인 제거** — `sample_lights`가 라이트 종류를 뽑을 때, 각 종류의 **이론적 최대 달성 면적**(해당 종류의 크기 계산에 모든 clamp를 적용한 값)을 계산해 목표 면적에 미치지 못하는 종류를 후보에서 제외하고, 남은 종류 중에서 균일하게 뽑는다. 위 표대로 목표가 작으면 세 종류 모두 후보에 남고, 커질수록 circle → beam 순으로 빠지며 최종적으로 rect만 남는다. 모든 종류가 제외되는 경우(rect조차 부족한 목표)는 `_MAX_LIGHTS_AREA_RATIO=0.60` 상한과 rect의 달성 능력(0.564)을 고려할 때 발생하지 않지만, 만약 발생하면 기존과 동일하게 `_require_exceeds`가 명확한 `ValueError`를 던지도록 둔다.

2. **방어선 추가** — `training/cli.py`의 배치 루프가 `ValueError`도 잡아서 해당 샘플만 건너뛰고 계속 진행하도록 한다(현재 `ClipTooShortError`만 처리). 1번으로 실패 자체가 사라지더라도, 향후 다른 이유로 발생할 수 있는 유사 실패가 전체 배치를 죽이는 것을 막는다. 건너뛴 샘플은 로그로 남긴다.

## 4. 밝기/색상 대비 다양화

### 4.1 general 패턴

현재 (training/params.py:200-201):
```python
dark_target = profile.general_flash_dark_threshold * 0.5
bright_target = min(1.0, dark_target + profile.general_flash_delta_threshold + 0.1)
```

변경:
- `dark_target`: `[0.0, profile.general_flash_dark_threshold * 0.5]` 균일 샘플링
- `bright_target`: `[dark_target + profile.general_flash_delta_threshold + 0.1, 1.0]` 균일 샘플링

현재 나오던 값이 각 구간의 한쪽 끝(dark는 최댓값, bright는 최솟값)이므로 기존 동작은 새 분포에 포함된다. 즉 이 변경은 기존 범위를 좁히지 않고 넓히기만 한다.

### 4.2 red 패턴

현재 (training/params.py:209-220):
```python
target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
r = 0.9
gb_sum = r / target_ratio - r
red_rgb = (r, gb_sum / 2, gb_sum / 2)
baseline_rgb = (0.3, 0.3, 0.3)
```

변경:
- `target_ratio`: `[profile.red_saturation_ratio_threshold + _RATIO_MARGIN, _MAX_SATURATION_TARGET]` 균일 샘플링
- `r`: `[0.5, _MAX_SATURATION_TARGET]` 균일 샘플링
- `baseline_rgb`: `[0.0, 0.5]`에서 값 하나를 뽑아 `(v, v, v)`로 구성
- `gb_sum` 계산식은 그대로 유지 (샘플링된 `r`과 `target_ratio`로 계산)

`r`의 하한을 0.5로 두는 이유: 그보다 낮으면 `gb_sum` 계산 결과가 지나치게 작아져 실질적인 빨간 플래시를 만들지 못한다. `baseline_rgb` 상한을 0.5로 두는 이유: 베이스라인이 빨간 펄스만큼 밝아지면 대비 자체가 사라진다.

### 4.3 안전장치 유지

기존 `_require_exceeds` 검증은 그대로 둔다. 어떤 값이 뽑히든 프로파일 임계값을 실제로 초과하는지 매번 확인하므로, 랜덤화 때문에 "위험하지 않은 샘플"이 학습 데이터에 섞이는 일은 없다. `synthesize_sample*`의 Detector 재검증 retry 루프도 그대로 유지된다.

## 5. 에러 처리

- 샘플링된 값이 `_require_exceeds`를 통과하지 못하면 기존과 동일하게 `ValueError`를 던진다(설정 오류를 조용히 넘기지 않는다는 기존 원칙 유지). 단 §3.3-2의 배치 루프 방어선 덕분에 배치 전체가 중단되지는 않는다.
- Detector 재검증 실패는 기존 retry 루프가 처리한다(`max_retries` 소진 시 `None` 반환, 해당 샘플 건너뜀).

## 6. 테스트 전략

**단위 테스트** (각 샘플링 함수당):
- 반복 호출(200회) 시 뽑힌 값이 항상 의도한 구간 안에 있다
- 반복 호출 시 서로 다른 값이 나온다(고정값이 아니다) — 고유값 개수로 검증
- 어떤 값이 뽑혀도 프로파일 임계값을 초과한다

**회귀 테스트**:
- 큰 목표 면적(0.60 부근)으로 `sample_lights`를 반복 호출해도 `ValueError`가 발생하지 않는다 — 달성 불가 종류 제외 로직 검증 (이전 실측 실패율 33% → 0%)
- 목표 면적이 작을 때는 세 종류(rect/circle/beam)가 모두 후보로 남아 실제로 다양하게 뽑힌다 — 제외 로직이 과하게 작동해 항상 rect만 나오는 퇴행을 막는다
- `training/cli.py` 배치 루프가 `ValueError`를 만나도 해당 샘플만 건너뛰고 계속 진행한다

**통합 검증**:
- 기존 테스트 전체(213개) 통과
- `data/synthetic` 전체 재합성 후, 마스크 면적과 대비 값의 실제 분포를 측정해 이전보다 넓어졌는지 확인
- 재합성 시 Detector 재검증 수용률을 실측해 이전(144/144 수준)보다 크게 떨어지지 않았는지 확인

## 7. 성공 기준

이 작업 자체의 완료 기준은 위 테스트 통과와 재합성 후 분포 확대 확인이다.

다만 **프로젝트 전체의 최종 목표는 별개**이며, 임의의 퍼센트 수치가 아니라 다음으로 정의한다: 보정된 영상을 `detector.pipeline.run_detection`에 다시 통과시켰을 때 위험 구간이 사라지거나 프로파일 임계값을 통과할 만큼 축소되는 것. 이번 데이터 다양화만으로 거기 도달하지 못할 가능성이 높으며(§2에서 제외한 지속시간·다중 구간·소스 도메인이 남아 있으므로), 재학습 후 이 기준으로 재측정해 다음 병목을 판단한다.
