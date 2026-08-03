# Realistic Flicker Injection — Design Spec

## 1. 배경과 목적

DatasetSynth의 기존 주입 함수(`inject_general_flash`/`inject_red_flash`, `training/injection.py`)는 위험 구간의 픽셀을 원래 장면 내용과 무관하게 **고정된 단색으로 덮어씌운다.** 이는 Detector가 확실히 위험으로 판정할 합성 데이터를 값싸게 만드는 데는 충분했지만(실제로 116개 샘플 생성에 성공), Mitigator 학습용으로는 비현실적이다 — 실제 PSE 위험(조명이 사람·사물을 비추며 반짝이는 것)은 장면의 질감·형태를 유지한 채 밝기/색만 변하는데, 지금 방식은 그 영역을 완전히 지워버려 Mitigator가 "납작한 색 패치를 원래대로 복원하는 법"만 배우게 될 위험이 있다.

**핵심 설계 원칙**: 픽셀을 고정값으로 대체하는 대신, **선형(linear) 공간에서 원본 밝기에 배율을 곱한다.** 이는 실제 조명 물리와 일치한다 — 광원의 세기가 배가되면 반사광 세기도 (선형 공간에서) 비례해서 변한다. sRGB 감마 압축된 값에 직접 배율을 곱하는 것은 물리적으로 부정확하다.

## 2. 스코프

**포함**:
- `_srgb_to_linear`/`_linear_to_srgb` 배열 헬퍼 (새로 추가, `training/injection.py`)
- `inject_general_flash_realistic(frames, window, gain_dark, gain_bright)` — 선형 공간 곱셈 배율, 어두운/밝은 상태 교대
- `inject_red_flash_realistic(frames, window, red_gains, baseline_gains)` — 채널별 배율(R 강조/G·B 억제) 교대. 두 상태 모두 명시적으로 조정된 값이며, 원본 색상이 이미 붉은 편이어도 안전하게 작동하도록 "기준 상태"도 R을 억제하는 배율을 적용 (원본 그대로 두지 않음)
- `synthesize_sample_realistic(clean_frames, fps, profile, pattern, rng, max_retries=5)` — `sample_synthesis_params`에서 `InjectionWindow`만 재사용하고, 고정 배율로 위 함수 호출. 검증/재시도 로직은 기존 `synthesize_sample`과 동일 (Detector 재확인, 최대 5회, `margin_seconds=0.0`)
- `training/cli.py`에 `--injection-mode {flat,realistic}` 플래그 추가 — 실제로 어느 쪽을 쓸지 배치 실행 시 고를 수 있어야 함

**제외 (변경하지 않음)**:
- 기존 `inject_general_flash`, `inject_red_flash`, `sample_synthesis_params` — 전혀 안 건드림. 기존 테스트·기존 생성 데이터의 유효성에 영향 없음
- 콘텐츠 적응형 배율 계산 (Approach B) — 실측 결과(실제 어두운 DAVIS 클립에서도 고정 배율 0.3/3.0이 필요 면적의 3배 이상 확보) 기준으로 불필요하다고 판단, 배제
- 마스크 모양 다양화 (사각형 → 복잡한 도형) — 별도 관심사, 이번 스코프 아님

## 3. 배율값 (실측 근거)

실제 DAVIS 클립(`bear.mp4`, 어두운 숲 장면, 영역 median 밝기 0.094)으로 검증:

| 배율 | 결과 flagged 면적 비율 |
|---|---|
| gain_dark=0.3, gain_bright=3.0 | 76.8% |
| gain_dark=0.5, gain_bright=2.0 | 61.3% |

가장 엄격한 프로파일(kr, 10%)과 나머지(25%) 모두 넉넉히 초과. 배율을 아무리 세게 줘도 threshold를 못 넘는 픽셀은 약 23%(원래 거의 검정)인데, 필요한 건 전체 면적의 10~35%뿐이라 실질적 문제 없음. 극단적으로 어두운 클립 전체는 재시도 5회 소진 → 기존 `validation_exhausted` 처리로 스킵.

기본값: `gain_dark=0.3`, `gain_bright=3.0`.

**적색 플래시는 훨씬 센 배율이 필요합니다 — 실측으로 발견.** 처음 시도한 `red_gains=(3.0, 0.3, 0.3)`은 실제로 **8%만 flagged**됐습니다. 원인: WCAG 채도비 테스트(`R/(R+G+B) >= 0.8`)는 sRGB(감마 압축된) 공간에서 계산되는데, 감마 압축은 값 차이를 눌러버리는 함수라 선형 공간의 10배 채널 비율 차이가 sRGB에서는 약 2.6배로 줄어듭니다. 재검증 결과:

| red_gains | baseline_gains | 랜덤 콘텐츠 flagged | 이미 붉은 콘텐츠 flagged | baseline 채도비 최대 |
|---|---|---|---|---|
| (3.0, 0.3, 0.3) | (1,1,1) | 8% | — | — |
| (20.0, 0.02, 0.02) | (1,1,1) | 95% | 84% | **0.858 (임계값 0.8 초과!)** |
| (20.0, 0.02, 0.02) | (0.3, 1.0, 1.0) | 95% | **100%** | 0.778 (안전) |

세 번째 줄이 최종 채택값: `red_gains=(20.0, 0.02, 0.02)`, `baseline_gains=(0.3, 1.0, 1.0)`. baseline에서 R을 억제하지 않으면(1,1,1), 원래 장면이 이미 붉은 편일 때 "기준 상태" 자체가 채도 임계값을 넘어버려서 XOR 기반 전환 감지가 깨집니다 — R 억제가 장식이 아니라 필수였습니다.

## 4. 데이터 흐름

```
clean_frames
  → sample_synthesis_params(profile, pattern, ...)  [기존 함수 그대로, InjectionWindow만 사용]
  → inject_general_flash_realistic / inject_red_flash_realistic (고정 배율)
       → 선형 변환 → 배율 곱함(상태 교대) → sRGB로 재변환
       → 결과: 원본 질감 보존, 밝기/채도만 변조된 degraded_frames
  → run_detection(degraded_frames, profile, margin_seconds=0.0)  [기존 검증 로직 그대로]
       → 통과: SynthesizedSample 반환 / 실패: 재시도(최대 5회) → 소진 시 skip
```

`training/cli.py`의 `--injection-mode` 플래그가 `synthesize_sample` vs `synthesize_sample_realistic` 중 어느 걸 `run_batch`가 호출할지 결정.

## 5. 에러 처리

- 배율 곱한 뒤 `np.clip(..., 0.0, 1.0)`으로 유효 범위 밖 값 방지
- 극단적으로 어두운 영역이 threshold를 못 넘는 것은 에러가 아닌 예상된 동작
- 재시도 5회 소진 시 `None` 반환 → 기존 `training/cli.py`의 `validation_exhausted` 처리 그대로 재사용 (변경 없음)

## 6. 테스트 전략

Detector·DatasetSynth와 동일한 스타일: 인메모리 합성 프레임, 실제 DAVIS 불필요. 단, 이번엔 **일부러 불균일한(균일하지 않은) 프레임**으로 테스트해서 질감 보존을 직접 검증한다.

- `inject_general_flash_realistic`: 윈도우/마스크 경계 테스트(기존과 동일 패턴) + **핵심 신규 테스트**: 마스크 영역 안에 서로 다른 두 밝기 값을 가진 프레임을 주입한 뒤에도 그 두 영역이 여전히 서로 다른 값을 유지하는지 확인 (flat 버전이라면 뭉개졌을 것) + 실제 `transition_mask`로 위험 판정되는지
- `inject_red_flash_realistic`: 같은 패턴 + `red_flash_mask` 검증
- `synthesize_sample_realistic`: 불균일한(그라디언트/노이즈) 합성 클립으로 통합테스트 — 재시도 루프가 텍스처 있는 콘텐츠에서도 작동하는지
- `training/cli.py`의 `--injection-mode realistic` 플래그가 실제로 realistic 함수를 호출하는지 라우팅 테스트

## 7. 열린 리스크 / 향후 로드맵

- **기존 `data/synthetic/`(flat 방식, 116개 샘플)의 처리**: Mitigator 학습에 안 쓸 계획이므로 삭제 예정(재생성 가능, git 미포함). 이 스펙 구현 후 realistic 모드로 DAVIS 재실행해서 대체 예정.
- **마스크 모양**: 여전히 사각형. Universal Access in HCI 학회 논문(Jordan, PSE 비디오 안전 도구 검증)에 따르면 탐지 알고리즘 검증엔 사각형이 표준적이나, Mitigator 학습 다양성 측면에서 나중에 재검토 필요할 수 있음.
- **배율값 재조정**: 지금 값(0.3/3.0)은 클립 1개(bear.mp4) 실측 기준. 90개 전체 재실행 후 성공률·재시도 분포를 보고 필요시 조정.
- **적색 플래시의 질감 보존은 실질적으로 없음 (수용된 한계)**: 리뷰 결과, realistic 모드의 적색 플래시 펄스 프레임은 사실상 flat한 포화 적색으로 나온다(R 채널 mean ~254, std ~0.38, 원본 장면과의 상관관계 ~0.083). 이는 버그가 아니라 3절에서 실측 후 채택한 배율(`red_gains=(20.0, 0.02, 0.02)`)의 필연적 결과다 — 감마 압축된 채도비 임계값(`R/(R+G+B) >= 0.8`)을 넘기려면 선형 공간에서 R을 충분히 크게 곱해야 하는데, 그 정도 배율은 시작 질감과 무관하게 R 채널을 255 근처로 밀어붙인다. 일반 플래시 경로는 질감 보존이 거의 완벽하다(상관관계 ~1.0)는 것과 대조적. 즉 적색 플래시 경로는 Detector 유효성(정확히 위험으로 플래그됨)은 달성하지만, 일반 플래시 경로와 같은 수준의 질감 보존은 달성하지 못한다 — 코드로 고칠 결함이 아니라 알려진 채택된 트레이드오프다.
