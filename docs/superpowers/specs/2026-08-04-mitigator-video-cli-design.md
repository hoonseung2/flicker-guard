# Mitigator 영상 단위 CLI — 설계 문서

## 1. 배경과 목적

`mitigator/infer.py`의 `mitigate_segment`는 이미 "위험 구간 판단 + 마스크 블렌딩 보정"을 프레임 시퀀스 단위로 수행하지만, 실제 영상 파일(`.mp4`)을 입력받아 보정된 영상 파일을 출력하는 사용자 대면 도구는 없다(`mitigator/infer.py` 자체가 "평가용, 런타임 파이프라인 연동 안 됨"이라고 명시). 학습된 Mitigator 체크포인트를 실제 데모 영상(예: 콘서트 조명 영상)에 눈으로 확인해보기 위한 최소 CLI가 필요하다.

## 2. 스코프

**포함**:
- `mitigator/cli.py` (신규) — `--input`, `--output`, `--checkpoint`, `--profile` 인자를 받는 CLI. `detector/cli.py`, `training/cli.py`와 동일한 스타일.
- 체크포인트에서 `MitigatorNet` 가중치만 로드하는 최소 로더(옵티마이저 상태는 필요 없음 — `training/train_mitigator.py`의 `load_checkpoint`는 옵티마이저를 요구하므로 재사용하지 않고 별도로 작성)
- 프레임을 다시 `.mp4`로 쓰는 헬퍼(`scripts/davis_to_clips.py`의 `cv2.VideoWriter` 패턴 재사용)

**제외 (범위 밖)**:
- 오디오 트랙 보존 — 결정됨(이번 스코프 아님), 영상(프레임)만 처리
- 실시간/스트리밍 처리, Verifier/Fallback 연동 — 이건 여전히 별도의 향후 로드맵. 이 CLI는 어디까지나 배치/평가용 도구
- 영상 자르기·구간 선택 UI — 전체 영상을 통째로 넣고 통째로 받는 구조

## 3. 동작 방식

```
mitigate_segment(frames, fps, profile, model)이 이미:
  - compute_prior로 프레임별 위험 여부 + 마스크 + target_histogram을 스스로 계산
  - target_histogram이 있는 위험 프레임만 마스크 안쪽을 보정
  - 그 외 프레임/픽셀은 원본 그대로 패스스루
을 전부 내장하고 있으므로, 이 CLI는 별도의 "위험 구간만 잘라서 넘기기" 로직 없이
영상 전체 프레임을 그대로 mitigate_segment에 통째로 넘기면 된다.
```

데이터 흐름:
```
detector.cli.read_video_frames(input_path) → frames, fps
load_profile(profile_path) → profile
torch.load(checkpoint_path)["model"] → MitigatorNet.load_state_dict(...)
mitigate_segment(frames, fps, profile, model) → corrected_frames
cv2.VideoWriter(output_path, fourcc, fps, (w, h)) → corrected_frames 각각 BGR uint8로 변환해 기록
```

## 4. 에러 처리

- 입력 영상 디코딩 실패: `detector.cli`의 기존 `VideoReadError`를 그대로 전파 (새로 만들지 않음)
- 체크포인트 파일 없음/형식 불일치: `torch.load`/`load_state_dict`가 자연히 던지는 예외를 그대로 전파 — 별도 래핑 불필요(사용자가 직접 실행하는 1회성 CLI라 스택트레이스로 충분)

## 5. 테스트 전략

- 작은 합성 프레임 시퀀스 + 랜덤 초기화 `MitigatorNet`으로 "체크포인트 저장 → 로드 → mitigate_segment 호출 → 영상 파일로 쓰기 → 다시 읽어서 프레임 수/해상도 확인"까지 엔드투엔드 유닛 테스트 (`tmp_path` 활용, 실제 학습된 가중치나 실제 영상 불필요)
- 이 스코프는 실제 트레인드 체크포인트가 아직 없어도(랜덤 가중치로도) 배관 자체를 지금 바로 검증 가능
