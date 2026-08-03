# flicker-guard

짧은 영상(광고·숏폼·클립)을 입력받아 **광과민성 발작(PSE) 위험 구간을 탐지하고, 그 구간만 실시간에 가깝게 완화**하는 파이프라인의 설계 문서(spec)입니다. 아직 코드는 없으며, 이 문서는 구현 계획(plan) 수립 전 단계의 설계 산출물입니다.

## 1. 배경과 상위 프로젝트와의 관계

이 프로젝트는 `All-In-One-Deflicker`(CVPR 2023, Layered Neural Atlases 기반 blind video deflickering)를 경량화·추론형으로 전환하려던 기존 작업에서 갈라져 나왔습니다. 분석 결과 두 가지가 확인되었습니다.

- `All-In-One-Deflicker`의 Stage 1(neural atlas)은 **영상 1개당 처음부터 최적화(test-time fitting)하는 구조**라, 아무리 하이퍼파라미터를 줄여도 "재생과 동시에 실시간 처리"라는 목표에는 근본적으로 맞지 않습니다.
- 팀이 실제로 원하는 형태는 **탐지(가벼움) + 완화(경량 causal 추론망)**로 분리된 파이프라인이며, 완화망의 아키텍처 아이디어는 두 논문에서 차용합니다.
  - **BlazeBVD** (arXiv 2403.06243, 코드 비공개): STE/히스토그램 기반 조명 prior로 학습 부담을 줄이는 아이디어
  - **Flickerformer** (`qulishen/Flickerformer`, CVPR 2026, 코드 공개): 3프레임 윈도우 → Restormer식 경량 U-Net 아키텍처. 단, 원래 목적은 burst 사진의 row-wise 밴딩 제거라 우리 문제(전역 조명 flicker)와는 다름 — **아키텍처만 참고**하고 PSE 데이터로 새로 학습.

시각피로·영상유발멀미(VIMS)는 이 프로젝트의 상위 서비스가 다루는 다른 축이지만, 이번 설계에는 포함하지 않습니다 (아래 6번 스코프 참고).

## 2. 규제 기준 요약 (탐지 로직의 근거)

| 기준/국가 | 임계 조건 |
|---|---|
| W3C | 초당 3회 이상 번쩍이는 콘텐츠 제한 |
| Netflix 가이드 | 초당 3회 이상 밝기 변화, 강한 적색 플래시, 5초 이상 지속 플래시 |
| 한국 | W3C 기준 + 3~50Hz 플래시가 화면 면적 10% 초과 금지, 3초 미만 반짝임 제한 |
| 일본 | 화면 면적 25% + 고대비/고밀도 패턴의 빠른 움직임 제한 |
| ITU | 화면 면적 25%, 휘도차 20 cd/m², 플래시 간 최소 프레임 간격(50Hz 환경 9프레임/60Hz 10프레임) |
| 영국 Ofcom (강제) | 화면 면적 25%, 3회 이상 동시 발생 시 9프레임 이상 간격도 규제, 어두운 장면 휘도 160 미만 + 명암차 20 이상 시 추가 규제 |
| 미국 | 10~20Hz 대역이 가장 위험 (강제 표준은 약함) |

탐지 파이프라인 구조는 **ITU-R BT.1702-04 (Fig.4)**의 신호처리 프레임워크를 참조 구조로 삼습니다: 디인터레이스 → RGB 변환 → 감마 보정 → 공간 필터(노이즈/미세패턴 오탐 저감) → 모션 추정/보상(패닝 오탐 저감) → 플리커 측정 → 임계값 적용. **딥러닝이 아닌 classical 신호처리**입니다.

## 3. 아키텍처 개요

```
[재생 버퍼]
    │
    ▼
Detector (classical, 항상 실행) ── 프레임별 위험도 측정 + 위험 구간(±마진) 산출
    │
    ├─ 안전 → 그대로 통과
    │
    └─ 위험 → PriorCalc(조명 prior) → Mitigator(Tier1, 학습된 경량 causal 망)
                                          │
                                          ▼
                                     Verifier (Detector 재사용으로 재검증)
                                          │
                                ┌─────────┴─────────┐
                               통과                  실패
                                │                     ▼
                                │            Fallback (Tier0: classical STE →
                                │             강도 상향 → 최후수단: 감광/프레임 홀드)
                                ▼                     ▼
                          [처리된 구간을 원래 타임라인에 splice-back]
                                          │
                                          ▼
                                     [출력 재생]
```

**핵심 원칙**
1. 탐지와 완화는 완전히 분리된 모듈 — 탐지기는 독립적으로 테스트·조정 가능
2. 안전 판정된 프레임은 신경망을 거치지 않음 (화질 보존 + 연산 절약)
3. 완화 모델을 100% 신뢰하지 않고, 처리 후 항상 재검증 → 실패 시 확실히 안전한 classical 방법으로 강등
4. 지연 예산: **구간당 3~5초 버퍼 허용** (팀 확인 완료). 완전한 프레임 단위 실시간은 목표가 아님

## 4. 컴포넌트

### 런타임 (항상 동작)

| 컴포넌트 | 책임 | 의존성 |
|---|---|---|
| Detector | classical 신호처리로 프레임별 위험도 측정, 위험 구간 산출 | 없음 (config 프로파일만) |
| PriorCalc | STE/히스토그램 기반 조명 prior 계산 (BlazeBVD 아이디어), Detector와 로직 일부 공유 가능 | 없음 |
| BufferManager | 재생 버퍼 관리, 안전/위험 라우팅, splice-back | Detector 구간 경계, 지연 예산 |
| Mitigator (Tier1) | 위험 구간 + prior → 보정된 프레임 (Flickerformer 백본 + prior conditioning) | 오프라인 학습된 가중치. 배포 시엔 순수 추론, 영상별 재학습 없음 |
| Verifier | 완화 결과를 Detector로 재검증 | Detector |
| Fallback (Tier0) | classical STE 보정 + 최후수단 | 없음 |

### 오프라인 (배포 전 1회, 클라우드 GPU로 학습)

| 컴포넌트 | 책임 |
|---|---|
| DatasetSynth | 2절의 규제 기준(주파수대역·면적%·휘도차)을 합성 파라미터로 삼아 clean 영상(DAVIS 등)에 PSE 패턴 flicker 합성, 생성된 라벨을 Detector로 재검증해 학습셋 채택 |
| TrainMitigator | Mitigator 학습 (Flickerformer 백본 + PriorCalc conditioning) |

## 5. 폴더 구조 (제안)

```
flicker-guard/
├── README.md
├── configs/
│   ├── profiles/            # kr.json jp.json itu.json ofcom.json w3c.json netflix.json
│   └── pipeline.json         # BufferManager 지연 예산 등
├── detector/                 # classical, 독립 테스트 가능
├── prior/
├── mitigator/
│   ├── arch.py
│   ├── infer.py
│   └── weights/
├── fallback/
├── buffer/
├── training/                 # 런타임과 분리된 오프라인 파이프라인
│   ├── dataset_synth.py
│   └── train_mitigator.py
├── tests/
└── reference/                 # BlazeBVD/Flickerformer 참고 노트, 라이선스 확인 메모
```

## 6. 스코프

**포함**: PSE(광과민성 발작) 위험 탐지 + 완화, 짧은 영상(수 초~수십 초) 기준, 위 6개국/기관 기준 프로파일화

**제외** (이번 설계 범위 아님): 시각피로·시각불편, 영상유발 멀미(VIMS) — 별도 착수 시 따로 설계

## 7. 에러 핸들링 / 에스컬레이션

| 단계 | 트리거 | 조치 |
|---|---|---|
| Level 0 | Mitigator 성공 + Verifier 통과 | 그대로 사용 |
| Level 1 | Mitigator 예외/NaN, 또는 지연 예산 초과 | Tier0 classical STE로 대체 → 재검증 |
| Level 2 | STE 보정 후에도 Verifier 실패 | 보정 강도 상향 → 재검증 |
| Level 3 (최후수단) | Level 2도 실패 | 마지막 안전 프레임 홀드 또는 강제 감광, 인시던트 로깅 |

추가 경계 조건:
- 버퍼 시작/끝 등 프레임 부족으로 측정이 불확실한 구간은 **보수적으로 위험 취급** (false negative가 가장 치명적인 실패 모드)
- 디코딩 실패/손상 프레임은 안전으로 간주하지 않고 Fallback 경로로 처리
- 모든 에스컬레이션 이벤트는 구간 타임스탬프·전후 Detector 측정값·도달 레벨을 로깅 (외부 검증 대조용)

## 8. 테스트 전략

| 대상 | 방법 |
|---|---|
| Detector | 임계값 경계값 유닛테스트 (예: 2.9Hz vs 3.0Hz, 면적 9.9% vs 10.0%) — 결정론적이라 GPU 불필요 |
| DatasetSynth ↔ Detector | 합성 flicker가 의도한 조건으로 실제 Detector에 걸리는지 회귀 테스트 |
| Mitigator | (a) PSNR/SSIM vs synthetic GT (b) **완화 후 Verifier 재통과율** — 이 프로젝트의 실질적 성공 기준 |
| End-to-end | 라벨링된 테스트 클립 (위험구간이 시작/끝에 걸치는 경우, 연속 위험구간, 씬 전환 겹침 등 경계 케이스 포함) |
| 지연 성능 | 목표 하드웨어에서 구간당 처리 시간 실측, 3~5초 예산 대비 마진 확인 |
| 폴백 드릴 | Mitigator 예외/타임아웃/재검증 실패를 의도적으로 주입해 에스컬레이션 사다리 검증 |

## 9. 오픈 리스크 / 확인 필요 사항

- **Flickerformer 코드 라이선스**: 리포에 LICENSE 파일이 없음. 아키텍처 아이디어 참고 수준을 넘어 코드를 직접 재사용하려면 별도 확인 필요.
- **PSE 특화 학습 데이터 부재**: 기존 공개 데이터셋(Deflicker의 Blind-Video-Deflickering-Dataset, Flickerformer의 BurstDeflicker) 모두 목적이 달라 그대로 못 씀 — DatasetSynth를 직접 구현해야 함.
- **외부 검증 필수**: 이 시스템은 의료기기·임상 진단 도구가 아니며, 자체 Detector가 학습 라벨링과 런타임 판정을 모두 담당하는 구조라 순환 검증 위험이 있음. **실제 서비스 배포 전 Harding FPA 등 검증된 외부 도구·전문가 검토가 반드시 필요**.
- **근거 수준 구분**: PSE는 법적 강제 표준이 있는 영역이나, 이 필터가 모든 경우에 발작을 예방한다고 단정할 수 없음 — 안전을 보장하는 표현은 사용하지 않는다.
- **WCAG 면적 기준의 "10도 시야각 서브셋" 미반영 (확인 완료, 근본적으로 어려움 — 당분간 보류)**: Jordan, "Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy" (Universal Access in HCI 학회 논문)에 따르면 WCAG의 면적 기준은 "화면 전체의 25%"가 아니라 "10도 시야각 서브셋 안에서의 25%"이며, 컴퓨터 화면 기준 약 416×416px(CSS 기준 픽셀)가 10도 시야각과 유사한 면적으로 제시됨. 현재 Detector의 `flagged_area_ratio`는 화면 전체 대비 비율로만 계산함. 관련 참고 자료: [traceRERC/pse-test-media](https://github.com/traceRERC/pse-test-media) (BSD-3-Clause) — `area_patterns/25pct_416x416`가 정확히 이 기준의 테스트 패턴을 제공.
  - **왜 지금 안 고치는지**: "10도 시야각"은 시청 거리 + 실제 화면 물리적 크기에 의존하는 개념. `pse-test-media`는 CSS 기준 픽셀(웹 표준상 가정된 시청 거리)로 우회하지만, 이는 웹 브라우저 컨텍스트에서만 성립. Detector는 임의 해상도 비디오 픽셀 배열만 받아 처리하므로, 실제 시청 환경(화면 크기·거리) 정보 없이는 "이 영상에서 몇 픽셀이 10도인지" 보편적으로 정의할 수 없음 — 임의 가정을 넣으면 부정확함을 감추는 꼴이 됨.
  - PriorCalc/Mitigator는 이 gap에 영향받지 않음 — Detector의 위험 판정을 그대로 재사용하는 구조라, 나중에 이 부분이 개선되면 코드 변경 없이 자동으로 물려받음.

## 10. 다음 단계

이 설계가 승인되면 `writing-plans` 단계로 넘어가 구현 계획을 수립합니다. 우선순위 후보:
1. Detector (classical, 의존성 적음) 먼저 구현·검증
2. DatasetSynth + Detector 일관성 확보
3. Mitigator 학습 (클라우드 GPU)
4. BufferManager + Fallback 통합, end-to-end 테스트

## 참고 자료

- All-In-One-Deflicker (CVPR 2023): https://github.com/ChenyangLEI/All-In-One-Deflicker
- BlazeBVD (arXiv:2403.06243): https://arxiv.org/abs/2403.06243
- Flickerformer (CVPR 2026): https://github.com/qulishen/Flickerformer
