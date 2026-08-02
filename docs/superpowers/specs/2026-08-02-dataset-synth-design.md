# DatasetSynth — Design Spec

## 1. 배경과 목적

`flicker-guard`의 로드맵(README §10)은 Detector(완료, `master`에 병합됨) 다음 단계로 DatasetSynth를 지정한다. Mitigator(Tier1 완화망)는 지도학습으로 "위험 구간이 포함된 입력 → 안전하게 보정된 출력"을 배워야 하는데, 목적에 맞는 공개 PSE 데이터셋이 없다(README §9: 기존 Deflicker/Flickerformer 데이터셋은 목적이 달라 그대로 못 씀). DatasetSynth는 clean 영상에 규제 기준을 넘는 flicker 패턴을 인위적으로 합성하고, Detector 자신으로 그 라벨을 재검증하여 학습셋을 만드는 오프라인 파이프라인이다.

**핵심 설계 원칙**: Detector가 실제로 측정하는 신호 모델(밝기 delta, dark threshold, 면적 마스크 비율, 적색 saturation ratio, 윈도우 주파수)을 합성 파라미터로 그대로 재사용한다. "왜 위험한지"를 아는 상태로 만들기 때문에, Detector 재검증은 맹목적 시행착오가 아니라 확인 절차에 가깝다.

**주파수 파라미터의 정확한 의미**: `ThresholdProfile.max_flashes_per_second`는 `FlickerScore.flagged_frame_count_last_second`(윈도우 내 flagged 프레임 개수, 시각적 flash rate의 약 2배 — I2 수정 이후의 실제 단위)와 비교된다. 따라서 합성 파라미터의 "주파수"도 시각적 flash rate가 아니라 **윈도우 내 flagged 프레임 개수 기준**으로 `max_flashes_per_second`를 넘도록 계산한다 — Detector가 실제로 비교하는 것과 동일한 단위를 맞추기 위함.

## 2. 스코프

**포함**:
- 합성 엔진: clean 클립의 한 구간에 general flash / red flash를 주입 (numpy/opencv, 새 의존성 없음)
- 6개 규제 프로파일(kr/jp/itu/ofcom/w3c/netflix) 각각의 `ThresholdProfile` 값을 넘는 파라미터 샘플링 — **프로파일마다 개별적으로** 계산하며, 전체를 아우르는 단일 min/max 범위는 쓰지 않는다
- Detector 재검증 + 검증 실패 시 재시도(최대 5회) → 소진 시 skip
- 결과물(clean/degraded 프레임 쌍 + 메타데이터) 저장, 재실행 시 이미 만든 샘플은 skip

**제외 (향후 로드맵으로만 명시, 이번 구현 범위 아님)**:
- DAVIS 데이터셋 다운로드 자동화 — 1회성 수동 데이터 준비 단계 (문서화만, 코드 없음)
- Approach B: 물리 기반 스트로브/글레어 시뮬레이션 — 더 사실적이지만 구현 복잡도가 높고 임계값을 확실히 넘긴다는 보장이 약함. Approach A(신호 모델 재사용)를 먼저 구현하고, 필요해지면 별도 설계로 착수
- 실제 DAVIS 클립으로 대량 데이터셋을 생성하는 실행 자체 — 파이프라인이 완성된 뒤의 별도 운영 작업 (테스트 스위트는 실제 DAVIS 없이 인메모리 합성 프레임만으로 CI에서 검증됨)
- TrainMitigator — 별도 계획

## 3. 모듈 위치

```
flicker-guard/
├── training/
│   ├── dataset_synth.py       # 이번 설계의 구현 대상
│   └── tests/
│       └── test_dataset_synth.py
```

(README §5 폴더 구조의 `training/` 하위, `dataset_synth.py`)

## 4. 컴포넌트

| 함수 | 역할 |
|---|---|
| `sample_synthesis_params(profile, pattern, clip_frame_count, fps, rng)` | 대상 `ThresholdProfile`의 임계값을 여유 있게(경계값 바로 위가 아니라 마진을 두고) 넘는 파라미터(주입 구간 `[start_frame, end_frame]`, 공간 마스크, 주파수, delta/saturation 강도)를 샘플링. 공간 마스크는 프레임 중앙을 기준으로 한 **직사각형 영역**이며, 그 크기는 `profile.max_area_ratio`를 넘도록 역산해서 정한다 (모양을 고정해야 목표 면적 비율을 정확히 계산할 수 있음) |
| `inject_general_flash(frames, window, area_mask, dark_target, delta_target, frequency)` | 지정 구간·마스크 안의 상대 휘도를 목표 주파수로 펄스(짝수/홀수 프레임 번갈아 베이스라인↔목표 휘도). 구간 밖 프레임/픽셀은 원본 그대로 유지 |
| `inject_red_flash(frames, window, area_mask, saturation_target, frequency)` | 같은 방식으로 채도 높은 빨강을 마스크 영역에 번갈아 오버레이 |
| `synthesize_sample(clean_frames, fps, profile, pattern, rng, max_retries=5)` | 파라미터 샘플 → 주입 → `detector.pipeline.run_detection(degraded_frames, fps, profile)` → 겹침 검증 → 통과 시 `SynthesizedSample` 반환, 실패 시 재샘플링 후 재시도, 소진 시 `None` |
| `write_sample(sample, out_dir)` | `clean/`, `degraded/` 프레임과 `meta.json`(원본 클립 id, 프로파일명, 패턴, 주입 구간, 사용 파라미터, Detector가 실제 반환한 scores/segments) 저장 |
| `main()` (CLI) | 입력 clean 클립 디렉터리 × 6개 프로파일 × 2개 패턴(× 조합당 N개 샘플)을 순회하며 배치 생성. 이미 존재하는 샘플은 skip(재실행 가능). 마지막에 프로파일×패턴별 성공/스킵 카운트 요약 출력 |

## 5. 데이터 흐름

```
clean clip (frames, fps)
  → (profile, pattern) 조합마다:
       sample_synthesis_params(profile, pattern, ...)
         → inject_general_flash / inject_red_flash → degraded_frames
         → run_detection(degraded_frames, fps, profile)
         → 검증: 반환된 RiskSegment 중 하나가 주입 구간을 포함하는가?
              (segment.start_frame <= window.start_frame
               and segment.end_frame >= window.end_frame)
         → 통과: write_sample(sample, out_dir)
         → 실패: 파라미터 재샘플링 후 재시도 (최대 5회) → 소진 시 skip, 실패 로그에 기록
  → 배치 종료 후 요약 리포트
```

주입 구간 밖은 손대지 않으므로 clean/degraded가 프레임 단위로 자동 정렬되고, Mitigator가 배울 정답(어느 프레임·어느 픽셀이 바뀌었는지)이 별도 계산 없이 그대로 나온다.

## 6. 에러 처리

- **원본 클립 읽기 실패** (손상/디코딩 불가): 해당 클립만 skip하고 로그, 배치는 계속 진행 (Detector CLI의 단일 영상 fail-loud 원칙과 달리, 여기는 다수 클립을 도는 배치 작업이므로 한 클립 때문에 전체가 멈추지 않음)
- **검증 재시도 소진**: 해당 (클립, 프로파일, 패턴) 조합은 skip, 마지막 시도 파라미터와 함께 실패 로그에 기록 (사후 디버깅용)
- **프로파일 JSON 자체가 깨졌거나 없음**: 배치 시작 전 설정 오류이므로 즉시 fail-fast
- **재실행(resume)**: 출력 디렉터리에 동일 id 샘플이 이미 있으면 기본적으로 skip. `--overwrite` 플래그로 강제 재생성
- **구조적 안전장치**: Detector 검증을 통과하지 못한 샘플이 저장되는 경로는 존재하지 않는다 — accept 경로는 검증 통과 지점 하나뿐

## 7. 테스트 전략

Detector와 동일한 스타일: 작은 인메모리 합성 프레임만으로 CI에서 결정론적으로 전부 통과해야 한다. 실제 DAVIS 클립은 테스트에 쓰지 않는다.

- `inject_general_flash`/`inject_red_flash` 유닛테스트: 작은 프레임에 주입 후, 같은 threshold 값으로 `detector.flash.transition_mask`/`red_flash_mask`가 실제로 그 구간을 flag하는지 확인
- `sample_synthesis_params` 유닛테스트: 반환된 파라미터가 대상 프로파일의 임계값을 (마진을 두고) 수치상 넘는지 검증 — 프로파일별로 파라미터화(parametrize)하여 6개 프로파일 모두 커버
- `synthesize_sample` 통합테스트: 작은 in-memory clean 클립 + 프로파일 하나로 정상 케이스가 accept되고 메타데이터가 올바른지 확인
- 재시도 소진 테스트: 검증이 항상 실패하도록 강제하여 정확히 `max_retries`번 시도 후 `None`을 반환하는지 확인
- `write_sample` 테스트: `tmp_path`에 써보고 디렉터리 구조/`meta.json` 필드 확인
- 재실행(resume) 테스트: 같은 출력 디렉터리로 두 번 실행 시 이미 있는 샘플은 다시 만들지 않는지 확인

## 8. 열린 리스크 / 향후 로드맵

- **Approach B (물리 기반 시뮬레이션)**: 이번 범위 아님. Approach A로 만든 데이터로 Mitigator를 학습해본 뒤, 실제 화면(카메라로 촬영된 스트로브 등)과의 도메인 갭이 문제가 되면 별도 설계로 착수
- **DAVIS 다운로드/실제 대량 생성 실행**: 파이프라인 완성 후의 운영 작업. 디스크 용량, 클립당 샘플 수, 커버리지 목표(프로파일×패턴 조합별 최소 샘플 수) 등은 이 실행 단계에서 별도로 결정
- **겹침 검증 기준의 엄격도**: 현재 기준(반환 segment가 주입 구간을 완전히 포함)은 margin 확장 덕분에 대체로 관대하지만, 실제 합성 결과에서 재시도율이 지나치게 높으면(예: 5회 재시도로도 자주 실패) 파라미터 샘플링의 마진을 넓히거나 검증 기준을 조정해야 할 수 있음 — 구현 초기 실측 후 확인 필요
