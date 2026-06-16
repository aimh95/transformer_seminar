# Transformer Seminar HTML

## 목적

LLM/Transformer 이해도가 낮은 팀원도 Transformer Encoder-Decoder 구조를 이해할 수 있도록 만든 세미나용 HTML 자료입니다.

## 주요 내용

- Tokenizer
- Embedding
- Positional Encoding
- Encoder
- Multi-Head Attention
- Q/K/V
- Scaled Dot-Product Attention
- FFN
- Decoder
- Masked Self-Attention
- Cross-Attention
- Linear + Softmax
- Training vs Inference

## 참고 자료

- Attention Is All You Need, Vaswani et al., 2017
- arXiv:1706.03762

## 실행 방법

```bash
# 방법 1: 브라우저에서 직접 열기
open index.html          # macOS
xdg-open index.html      # Linux

# 방법 2: 로컬 서버 (SVG 로딩 오류 시)
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080 접속
```

## 산출물

```
transformer_seminar_html/
  index.html                          # 메인 21섹션 세미나 HTML (섹션 0~20)
  README.md
  preview.png
  assets/
    01_overall_flow.svg               # 전체 Encoder-Decoder 아키텍처
    02_tokenizer_embedding_position.svg  # Tokenizer → Embedding + PE
    03_encoder_layer.svg              # Encoder Layer 내부 구조 (× 6)
    04_qkv_projection.svg             # Q/K/V 생성 (W^Q, W^K, W^V)
    05_attention_score_matrix.svg     # 6×6 Attention Score 히트맵
    06_scaled_dot_product_attention.svg  # Attention 4단계 파이프라인
    07_multi_head_attention.svg       # Multi-Head Attention (h=8)
    08_ffn_position_wise.svg          # Position-wise FFN
    09_decoder_shifted_target.svg     # Decoder Shifted Target 입력
    10_look_ahead_mask.svg            # Look-Ahead Mask 행렬
    11_cross_attention.svg            # Cross-Attention 흐름
    12_training_vs_inference.svg      # 학습 vs 추론 비교
```

## 섹션 구성

| 섹션 | 내용 | Nav |
|------|------|-----|
| 0 | 전체 흐름 한눈에 보기 | 전체 흐름 |
| 1 | Transformer는 왜 나왔는가? | |
| 2 | Tokenizer | Tokenizer/Embedding |
| 3 | Embedding | |
| 4 | Positional Encoding | |
| 5 | Encoder-Decoder 전체 구조 | Encoder |
| 6 | Encoder Layer 구조 | |
| 7 | Q, K, V 생성 | Attention |
| 8 | Attention Score Matrix | |
| 9 | Scaled Dot-Product Attention | |
| 10 | Multi-Head Attention | |
| 11 | Residual Connection과 Layer Normalization | |
| 12 | Feed Forward Network | |
| 13 | Decoder 구조 | Decoder |
| 14 | 학습 시 Decoder 입력: Shifted Target | |
| 15 | Look-Ahead Mask | Mask |
| 16 | Cross-Attention | |
| 17 | Padding Mask | |
| 18 | Linear + Softmax | Output |
| 19 | 학습 시점 vs 추론 시점 | 학습 vs 추론 |
| 20 | 전체 요약 | |

## Interactive Simulator

이 HTML에는 **Forward Pass부터 Backpropagation까지** 보여주는 toy Transformer 시뮬레이터가 포함되어 있습니다.

시뮬레이터는 실제 대형 모델의 수치를 재현하는 것이 아니라,
입력 token이 vector로 바뀌고, attention/FFN/decoder를 거쳐 확률과 loss가 만들어진 뒤,
gradient가 weight를 업데이트하는 방향을 시각적으로 이해하기 위한 교육용 예시입니다.

- toy example: d_model=4, d_k=3, vocab=7 (illustrative values)
- 14 step 구성: 입력 → Tokenizer → Embedding → PE → Q/K/V → Score → Softmax → AttOutput → FFN → Logits → Prob → Loss → Backprop
- Beginner / Detail 모드 전환
- 재생 / 일시정지 / 이전 / 다음 / 속도 조절

## 기술 정확성 체크리스트

- [x] Look-Ahead Mask는 Q/K/V가 아닌 Score Matrix에 적용 (MaskedScore = Score + Mask)
- [x] √d_k 스케일링 이유: d_k 차원의 분산 증가 (seq_len 때문이 아님)
- [x] Multi-Head는 d_model 차원을 head 수로 나눔 (시퀀스 분할 아님)
- [x] Q/K/V 역할은 설계가 아닌 학습의 결과
- [x] FFN은 token 간 정보를 새로 섞지 않음 (position-wise 독립 처리)
- [x] Residual: LayerNorm(x + SubLayer(x)), shape 동일 필요
- [x] Encoder × 6, Decoder × 6 (논문 기본값)
- [x] 학습: Teacher Forcing + 병렬, 추론: Autoregressive + 순차

## 예제 문장 (전 섹션 일관 사용)

```
Source: "The cat is on the mat."   → token ids: [1, 5, 2, 6, 1, 10]
Target: "Le chat est sur le tapis."
Decoder input: [<SOS>, Le, chat, est, sur, le]
```
