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

## 사전 지식 섹션

이 HTML은 선형대수나 신경망을 모르는 팀원도 Transformer 본론으로 들어갈 수 있도록,
Vector, Matrix, Weight, Dot Product, Softmax, Loss, Backpropagation의 최소 직관을 먼저 설명합니다.

이 섹션은 수학적 엄밀성을 목표로 하기보다,
Transformer의 Embedding, Q/K/V, Attention Score, Softmax, Loss 흐름을 이해하기 위한 준비 단계입니다.

### 사전 지식 섹션 구성 (8개 mini lesson)

| 번호 | 제목 | Q&A anchor |
|------|------|-----------|
| 0-1 | 숫자 하나로는 단어의 의미를 담기 어렵다 | — |
| 0-2 | Vector는 여러 특징을 담은 숫자 묶음이다 | vector-basic ★ |
| 0-3 | Matrix는 숫자가 들어 있는 표다 | matrix-basic |
| 0-4 | Weight는 모델이 학습하는 조절값이다 | weight-basic ★ |
| 0-5 | Vector에 weight matrix를 곱하면 새로운 표현이 된다 | matrix-multiplication-basic ★ |
| 0-6 | Dot product는 두 vector가 얼마나 잘 맞는지 계산한다 | dot-product-basic ★ |
| 0-7 | Softmax는 점수를 "참고 비율"처럼 바꾼다 | softmax-basic |
| 0-8 | Loss는 틀린 정도, Backpropagation은 고치는 방향이다 | loss-basic, backprop-basic ★ |

★ = seed Q&A 포함

## Q&A 주석 시스템

각 섹션 핵심 개념 아래에 접고 펼 수 있는 **Q&A 블록**이 삽입됩니다. 우측 하단 **Q&A 목록 패널**에서 전체 목록을 보고 클릭하면 해당 위치로 이동합니다.

### Q&A URL 직접 접근

```
index.html#qa-attention-score-matrix-001
index.html#qa-qkv-projection-001
index.html#qa-look-ahead-mask-001
index.html#qa-ffn-position-wise-001
index.html#qa-backpropagation-001
```

### 새 Q&A 추가 방법

1. `index.html` 내 `<script>` 블록의 `const qaItems = [...]` 배열에 항목 추가
2. 형식:
   ```javascript
   {
     anchor: 'look-ahead-mask',          // data-anchor 값 (기존 20개 중 선택)
     id: 'qa-look-ahead-mask-002',       // 고유 ID (anchor + 번호)
     q: '질문 텍스트',
     short: '한 줄 핵심 답변',
     detail: '상세 설명 (2~4문장)',
     summary: '한 줄 요약',
   },
   ```
3. 같은 `anchor`를 가진 항목이 여러 개면 같은 위치에 순서대로 삽입됩니다.
4. 새 anchor가 필요하면 해당 섹션 HTML에 `<div class="qa-target" data-anchor="새-이름"></div>` 추가.

### 현재 anchor 위치 (20개)

| anchor | 섹션 |
|--------|------|
| tokenizer | sec2 |
| embedding-lookup | sec3 |
| positional-encoding | sec4 |
| encoder-decoder-overview | sec5 |
| encoder-layer | sec6 |
| qkv-projection | sec7 ★ |
| attention-score-matrix | sec8 ★ |
| scaled-dot-product | sec9 |
| softmax-attention-weight | sec9 |
| value-weighted-sum | sec9 |
| multi-head-attention | sec10 |
| residual-layernorm | sec11 |
| ffn-position-wise | sec12 ★ |
| decoder-shifted-target | sec14 |
| look-ahead-mask | sec15 ★ |
| cross-attention | sec16 |
| padding-mask | sec17 |
| linear-softmax | sec18 |
| training-vs-inference | sec19 |
| backpropagation | 시뮬레이터 ★ |

★ = seed Q&A 포함

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
