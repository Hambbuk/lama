# PR – 썸네일 인페인팅 리팩터 & NPU 지원

## 개요
원본 **LaMa** 레포를 기반으로 썸네일 인페인팅 워크플로를 심플하게 재구성하고,
• 손(Hand) 마스크를 활용한 학습 파이프라인 추가  
• Verisilicon **VIPNano-Ql.7120** NPU 변환 오류를 해결하기 위한 커스텀 텐서 헬퍼 도입  
• ONNX 내보내기 및 테스트 유틸 추가  
등을 포함합니다.

## 주요 변경 사항
1. **레포 베이스**  
   - 원본 LaMa: <https://github.com/advimman/lama>  
   - ONNX Export 참고: Carve-Photos 포크 <https://github.com/Carve-Photos/lama>

2. **손 마스크 학습 지원**  
   - **@dohoon19-kim** 님 기여.  
   - `train_hand_mask/` PNG을 제공하고 `kind: hand_mask[_multi]` 로 설정하면 네트워크가 손 영역을 인페인트하지 않습니다.

3. **NPU 변환 오류 패치**  
   A311D · VIPNano 변환 시 `torch.flip`, `torch.tensordot`, `torch.matmul` 등이 실패하여
   `saicinpainting/training/module/ffc.py` 에 다음과 같은 경량 구현을 추가했습니다.

```python
"""커스텀 텐서 헬퍼 — PyTorch 기본 함수 대신 사용"""

def flip_axis_static(x, axis):
    dim = axis if axis >=0 else x.ndim + axis
    rev = torch.arange(1, x.size(dim)-1, device=x.device).flip(0)
    return x.index_select(dim, rev)

# matmul / tensordot 는 F.linear 기반 구현 (코드 생략)
```

4. **프로젝트 정리**  
   - Conda 의존 제거 → **Python 3.10 + `pip -r requirements.txt`** 만으로 설치.  
   - Bash 스크립트(`train.sh`, `inference.sh`, `export_to_onnx.sh`, `onnx_inference.sh`) 추가.  
   - README를 최소 가이드로 재작성.

## 사용 방법
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 학습
./scripts/train.sh

# 추론
./scripts/inference.sh -m experiments/<run>

# ONNX 내보내기 + 간단 테스트
./scripts/export_to_onnx.sh -m experiments/<run> -c best.ckpt -o lama.onnx -t
./scripts/onnx_inference.sh -m lama.onnx
```

## 감사의 글
- 손 마스킹 파이프라인: **@dohoon19-kim**  
- 원본 LaMa 팀 및 Carve-Photos 팀  
- VIPNano-Ql.7120 NPU 테스트 장비 제공: **Verisilicon**