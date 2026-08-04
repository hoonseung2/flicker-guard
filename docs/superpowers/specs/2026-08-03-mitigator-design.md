# Mitigator — Design Spec

## 1. 배경과 목적

README 아키텍처(`Detector → PriorCalc(조명 prior) → Mitigator(Tier1)`)의 세 번째 단계이자, 로드맵상 "실제로 위험 프레임을 고치는" 유일한 컴포넌트다. Detector(classical, 위험 판정)와 PriorCalc(classical, "원래 밝기가 어땠어야 하는지"에 대한 힌트 계산)는 이미 구현되어 있고, Mitigator는 이 둘의 출력을 받아 실제로 프레임을 복원하는 신경망이다.

**왜 코드 기반 규칙(예: 밝기 강제로 낮추기)이 아니라 AI인가** (`docs/team-overview.md`에 이미 기록된 결론을 그대로 계승):
- 규칙 기반 보정은 "이 부분만 갑자기 이상해졌다"는 티가 나기 쉽다 — 목표는 눈치채지 못할 정도의 자연스러운 보정
- 고정 규칙은 실제 위험한 스트로브와 연출 의도(폭발·번개 등 자극적이지만 안전한 장면)를 구분하지 못한다
- 실제 위험 패턴은 깔끔한 도형이 아니라 장면과 함께 움직이는 불규칙한 광원 — 규칙보다 AI가 더 정교하게 대응

아키텍처 아이디어 출처는 README에 이미 명시됨: **Flickerformer**(`qulishen/Flickerformer`, 3프레임 윈도우 → Restormer식 경량 U-Net)의 구조만 참고한다. 코드는 리포에 LICENSE 파일이 없어 저작권법상 기본값(모든 권리 보유)이 적용되므로 복사하지 않고 처음부터 새로 구현한다 — 브레인스토밍 중 별도로 확인·결정됨. Flickerformer의 원래 목적(burst 사진의 row-wise 밴딩 제거)과 우리 문제(전역 조명 flicker)는 다르므로 어차피 입력 구조·conditioning·손실 함수를 새로 설계해야 하며, "3프레임 윈도우 + 경량 어텐션 병목"이라는 패턴 자체는 이미지 복원 분야에서 널리 쓰이는 일반적인 설계라 코드를 보지 않고도 재구현 가능하다.

또한 브레인스토밍 중 **BurstDeflicker**(`qulishen/BurstDeflicker`, Flickerformer의 벤치마크 데이터셋) 활용 가능성을 검토했으나 기각했다: 이 데이터셋의 "flicker"는 롤링 셔터 카메라와 AC 전원 조명 간섭으로 생기는 **행(row) 단위 밴딩 아티팩트**로, PSE의 **전역 밝기/색 급변**과는 물리적 원인이 완전히 다르다. 게다가 burst(연속 정지 이미지 몇 장) 단위 구조라 DatasetSynth가 요구하는 클립 길이 조건도 못 채울 가능성이 높다. DAVIS 90개 클립(현재 합성 샘플 170개) 기반 데이터로 계속 진행한다.

## 2. 스코프

**포함**:
- `MitigatorNet` 모델 아키텍처 (`mitigator/arch.py`)
- 학습 데이터 파이프라인 (`training/mitigator_dataset.py`) — DatasetSynth의 `data/synthetic/` 출력 + PriorCalc의 `compute_prior`를 엮어 학습 예시 생성
- 학습 루프 (`training/train_mitigator.py`)
- 최소 추론 wrapper (`mitigator/infer.py`) — 학습된 가중치로 위험 구간 하나를 복원 (결과 확인/평가용)
- `training/dataset_writer.py`에 `fps` 필드 추가 (기존 `meta.json`에 없어서 PriorCalc 재계산 시 필요 — 작은 하위호환 확장)

**제외 (향후 로드맵)**:
- Verifier(완화 결과 재검증), Fallback(classical STE 대체), BufferManager(재생 버퍼·splice-back) 통합 — README 로드맵상 별도의 다음 단계(#4)
- 실제 프로덕션 수준의 본 학습(수십 epoch 완주, 데이터 확장, 필요시 클라우드 GPU) — 이번 스코프의 "완료"는 **파이프라인이 올바르게 동작하는 것**(소규모 유닛테스트 + 소량 데이터 smoke test + 1 epoch 실측 타이밍)까지이며, 실제 다수 epoch 본 학습은 구현 완료 후 별도 수동 작업
- 시간적 일관성(temporal consistency) 손실 항 — 1차로는 L1+SSIM만 사용, 실제 학습 결과를 보고 필요 시 추가 여부 결정
- 마스크 모양 다양화(사각형 외 형태), 배율/증강 재조정 — DatasetSynth 스펙에 이미 열린 리스크로 기록됨, 여기서 재론하지 않음

## 3. 모듈 위치

```
flicker-guard/
├── mitigator/
│   ├── __init__.py
│   ├── arch.py              # MitigatorNet 정의 (학습·런타임 공용)
│   ├── infer.py             # 추론 wrapper
│   ├── weights/              # 학습된 가중치 (git 미포함, data/처럼 재생성 가능한 아티팩트)
│   └── tests/
│       ├── test_arch.py
│       └── test_infer.py
└── training/
    ├── mitigator_dataset.py  # 데이터 로딩 (오프라인 전용)
    ├── train_mitigator.py    # 학습 루프 (오프라인 전용)
    ├── dataset_writer.py     # 기존 파일 수정 (fps 필드 추가)
    └── tests/
        ├── test_mitigator_dataset.py
        └── test_train_mitigator.py
```

## 4. 컴포넌트

| 함수/클래스 | 위치 | 역할 |
|---|---|---|
| `MitigatorNet` | `mitigator/arch.py` | 3프레임 윈도우(t-1,t,t+1 degraded RGB) + 위험 마스크(1채널) + target_histogram(조건) → 프레임 t에 대한 잔차(delta) 예측. 경량 인코더-디코더 + 병목 어텐션 블록 |
| `mitigate_frame(window, mask, histogram, model) -> np.ndarray` | `mitigator/arch.py` 또는 `infer.py` | 모델 순전파 + 안전 블렌딩(`mask*restored + (1-mask)*degraded`)을 합친 헬퍼 |
| `mitigate_segment(frames, fps, profile, weights_path) -> list[np.ndarray]` | `mitigator/infer.py` | 위험 구간 프레임들을 받아 `prior.compute.compute_prior`로 조건을 계산하고, 프레임별로 `mitigate_frame`을 슬라이딩 적용. `target_histogram=None`인 프레임은 원본 그대로 통과 |
| `load_mitigator_samples(data_dir, split) -> Dataset` | `training/mitigator_dataset.py` | `data/synthetic/`의 각 샘플 디렉터리를 읽어 (clean, degraded, mask, target_histogram) 학습 예시 시퀀스로 변환. `prior_cache.npz` 캐싱, clip_id 해시 기반 train/val 분할, 경계 프레임 복제 |
| `train(args) -> None` | `training/train_mitigator.py` | AdamW + L1/SSIM 손실 학습 루프. 체크포인트 저장/재개, 에폭별 JSON/CSV 로그, val 지표 계산 |
| `write_sample(..., fps: float)` | `training/dataset_writer.py` (기존 파일 수정) | `meta.json`에 `fps` 필드 추가 |

## 5. 데이터 흐름

**학습 데이터 준비 (`training/mitigator_dataset.py`)**
```
data/synthetic/<sample_id>/  (clean/, degraded/, meta.json)
  → prior_cache.npz 있으면 로드, 없으면:
       compute_prior(degraded_frames, fps, profile) 실행 → 캐시에 저장
  → meta.json의 segments(탐지된 위험 구간)에 속하고 target_histogram != None인 프레임만 선택
  → 각 프레임 i에 대해 (degraded[i-1:i+2], mask[i], target_histogram[i]) → clean[i] 학습 페어 구성
     (경계 프레임은 이웃 복제)
  → clip_id 해시(zlib.crc32(clip_id) % 100 < 80)로 train/val 배정
```

**학습 (`training/train_mitigator.py`)**
```
mitigator_dataset → 랜덤 패치 크롭(예: 256x256) → 배치
  → MitigatorNet 순전파 → 잔차 → restored = degraded[t] + delta
  → loss = L1(restored, clean[t]) + λ*(1 - SSIM(restored, clean[t]))
  → 역전파, AdamW step (mixed precision)
  → 매 N 에폭: val 지표 계산 + 체크포인트 저장 (best.pt 갱신)
```

**추론 (`mitigator/infer.py`, 이번 스코프에서는 평가용으로만 사용)**
```
위험 구간 프레임들 (Detector가 이미 선정, 이번 스코프에선 val 샘플의 degraded_frames)
  → compute_prior로 프레임별 (target_histogram, mask) 계산
  → 프레임별 3프레임 윈도우 슬라이딩 + MitigatorNet 순전파
  → mask 블렌딩으로 안전 보장
  → 복원된 프레임 시퀀스 (PSNR/SSIM 비교, 육안 확인용)
```

## 6. 에러 처리

- **clean/degraded 프레임 수·해상도 불일치**: 데이터 로딩 시점 fail-fast (기존 `training/cli.py` 스타일과 동일)
- **아주 짧은 위험 구간(1~2프레임)**: 경계 프레임 복제로 자연히 3프레임을 채움, 별도 특수 케이스 불필요
- **추론 중 NaN/Inf 출력**: 해당 프레임은 원본 그대로 통과 (이상 출력을 내보내지 않는다는 보장까지만, 실제 폴백 정책은 스코프 밖)
- **학습 중 NaN loss**: 조용히 넘어가지 않고 명확한 에러/로그로 드러내고 학습 중단 (fail-loud — freeze-and-fallback이 아님)
- **`prior_cache.npz` 캐시 무효화**: DatasetSynth의 기존 관례(재실행 시 사람이 직접 삭제 후 재생성)를 그대로 따름, 자동 버전 태그는 과설계로 보고 넣지 않음
- **`target_histogram=None` 프레임**: 학습에서 제외, 추론에서 pass-through — 조건 정보 없이 모델에 넣지 않는다는 원칙을 양쪽에서 동일하게 적용

## 7. 테스트 전략

Detector·DatasetSynth·PriorCalc와 동일한 스타일: 작은 인메모리 합성 데이터로 CI에서 결정론적으로 검증. 실제 DAVIS 데이터·GPU 학습은 CI 대상이 아니며 구현 완료 후 수동으로 실행.

- `MitigatorNet`: 다양한 입력 크기에 대해 forward pass의 출력 shape이 입력과 일치하는지, 그래디언트가 흐르는지(backward 후 파라미터 grad가 None이 아닌지) 확인
- `mitigate_frame`: 마스크 블렌딩이 마스크 바깥 픽셀을 정확히 원본과 동일하게 유지하는지 (핵심 안전장치 검증)
- `mitigator_dataset`: 
  - `segments` 밖 프레임 또는 `target_histogram=None` 프레임이 학습 예시에서 제외되는지
  - clip_id 해시 분할이 결정론적이고, 같은 clip의 general/red 샘플이 항상 같은 split에 속하는지 (데이터 유출 방지 검증)
  - `prior_cache.npz`가 있으면 재계산 없이 로드되는지 (캐시 히트를 모킹으로 검증)
  - 경계 프레임(클립 첫/마지막 위험 프레임)에서 이웃 복제가 올바른지
- `train_mitigator`: 아주 작은 가짜 데이터셋(몇 프레임, 작은 해상도)으로 1~2 스텝만 돌려 loss가 유한하고 감소 방향인지, 체크포인트 저장/재개 후 상태가 정확히 복원되는지 (옵티마이저 상태 포함)
- `mitigate_segment`: 랜덤 초기화 모델로 윈도잉·마스크 블렌딩·None-히스토그램 통과 로직 검증 (실제 학습 가중치 불필요)
- **구현 완료 후 수동 검증** (자동 테스트 아님): 실제 170개 샘플로 1 epoch 실행해 소요 시간 실측 + 리포트

## 8. 열린 리스크 / 향후 로드맵

- **데이터 규모**: 170개 샘플(위험 구간만 추리면 실질적으로 더 적음)이 견고한 학습에 충분한지 미지수 — 실제 학습 곡선을 보고 DAVIS 소스 다양화나 증강 여부 판단
- **general/red 데이터 불균형**: 87(red) vs 83(general), 이미 알려진 값
- **red-flash 샘플은 학습이 더 어려울 가능성**: `2026-08-03-realistic-injection-design.md` §7에 기록된 대로 red-flash 펄스 프레임은 질감이 거의 남지 않은 flat 포화 상태라, 모델이 참고할 정보가 이웃 프레임과 prior뿐 — 실제 결과를 보고 판단
- **LFRM 스타일 텍스처 복원 (향후 로드맵)**: BlazeBVD의 LFRM(Local Flicker Removal Module — optical flow로 이웃 프레임에서 과/저노출 영역에 텍스처를 이식)은 이번 설계에 채택하지 않았음(§1, PriorCalc 설계 문서 참고 — Detector의 기존 신호를 재사용하는 쪽을 택함). 다만 바로 위 항목의 red-flash 텍스처 손실 문제는 구조적으로 LFRM이 정확히 겨냥하는 문제라, 실제 학습 결과에서 red-flash 복원 품질이 불충분한 것으로 확인되면 LFRM 스타일의 optical-flow 기반 이웃 프레임 텍스처 이식을 Mitigator에 추가하는 것을 향후 개선안으로 검토. 지금은 채택하지 않음 — 현재 구조(잔차 예측 + 마스크 블렌딩)로 충분한지 실측 후 판단.
- **하이퍼파라미터(학습률·패치 크기·에폭 수)는 잠정값**: smoke test + 1 epoch 실측 이후 조정 예정
- **본격적인 프로덕션 학습**: 에폭 대폭 증가·데이터 확장·필요시 클라우드 GPU는 이번 스코프 밖, 로컬 학습 실측 후 판단
- **WCAG 10도 시야각 서브셋 면적 기준** (README §9): Detector의 기존 한계이며 PriorCalc/Mitigator는 이 gap에 영향받지 않음 (Detector 판정을 그대로 재사용하는 구조라 나중에 개선되면 자동 상속) — 재론 불필요
