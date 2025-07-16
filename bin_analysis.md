# bin/ 디렉토리 파일 분석 결과

## 학습/인퍼런스에 필요한 파일들 (유지)

### 핵심 실행 파일들
- **train.py** - 모델 학습 메인 파일
- **predict.py** - 모델 인퍼런스 메인 파일
- **make_checkpoint.py** - 체크포인트 생성 파일
- **to_jit.py** - JIT 모델 변환 파일

### 데이터 전처리 파일들
- **gen_mask_dataset.py** - 마스크 데이터셋 생성
- **gen_mask_dataset_hydra.py** - Hydra 설정을 사용한 마스크 데이터셋 생성
- **gen_outpainting_dataset.py** - 아웃페인팅 데이터셋 생성
- **gen_debug_mask_dataset.py** - 디버그용 마스크 데이터셋 생성
- **filter_sharded_dataset.py** - 샤딩된 데이터셋 필터링
- **split_tar.py** - tar 파일 분할
- **extract_masks.py** - 마스크 추출

### 예시/템플릿 파일들
- **mask_example.py** - 마스크 예시 생성
- **evaluator_example.py** - 평가자 예시
- **predict_inner_features.py** - 내부 특징 예측

## 삭제 가능한 파일들 (연구/논문용)

### 분석/평가 파일들
- **analyze_errors.py** - 오류 분석 시각화 (17KB)
- **evaluate_predicts.py** - 예측 결과 평가 
- **blur_predicts.py** - 예측 결과 블러 처리
- **calc_dataset_stats.py** - 데이터셋 통계 계산
- **report_from_tb.py** - TensorBoard 리포트 생성
- **sample_from_dataset.py** - 데이터셋 샘플링
- **side_by_side.py** - 결과 비교 시각화

### 논문 관련 디렉토리들
- **paper_runfiles/** - 논문 실험 스크립트들 (11개 파일)
- **debug/** - 디버깅 스크립트들 (1개 파일)

## 권장사항

1. **즉시 삭제 가능**: `paper_runfiles/`, `debug/` 디렉토리 전체
2. **연구용 파일 삭제**: 위 "삭제 가능한 파일들" 리스트의 분석/평가 파일들
3. **용량 절약**: 총 약 50KB+ 절약 가능

이 파일들을 삭제하면 핵심 학습/인퍼런스 기능은 그대로 유지하면서 레포지토리 크기를 상당히 줄일 수 있습니다.