# Transformer 세미나 노트 수정본

## 개요

Transformer는 기존 RNN, CNN 기반 Sequence 모델이 가진 한계를 해결하기 위해 제안된 구조입니다.

기존 RNN 계열 모델은 문장을 순차적으로 처리하기 때문에 병렬화가 어렵고, 멀리 떨어진 token 간의 관계를 학습하는 데 한계가 있었습니다. CNN 기반 sequence 모델은 병렬 처리는 가능하지만, 멀리 떨어진 token 사이의 정보를 연결하려면 여러 layer를 거쳐야 합니다.

Transformer는 이러한 한계를 해결하기 위해 **Attention Mechanism**을 중심으로 설계되었습니다. Attention을 사용하면 문장 안의 token들이 서로를 직접 참고할 수 있습니다.

Transformer의 주요 특징은 다음과 같습니다.

1. **순차 처리 제거**  
   문장을 한 token씩 순서대로 처리하지 않고, 전체 sequence를 한 번에 입력받아 병렬 연산할 수 있습니다.

2. **병렬 연산 가능**  
   RNN보다 학습 속도와 연산 효율이 좋아집니다.

3. **장거리 의존성 학습 개선**  
   멀리 떨어진 token도 Attention을 통해 직접 연결될 수 있습니다.

4. **Recurrent 구조 제거**  
   RNN이나 LSTM 없이도 번역 성능을 크게 향상시켰습니다.

Transformer의 핵심 기여는 다음과 같이 정리할 수 있습니다.

> 문장을 순차적으로 읽지 않아도, Attention만으로 token 간 관계를 효과적으로 학습할 수 있다.

이번 세미나에서는 Transformer가 입력 문장을 어떻게 token, embedding, positional encoding, attention, feed forward network를 거쳐 처리하고, 최종적으로 번역 문장을 생성하는지 **데이터 흐름 중심**으로 살펴보겠습니다.

---

## 구조 설명

논문에서 소개하는 Transformer는 **Encoder-Decoder 구조**를 가진 번역 모델입니다.

예를 들어 아래 문장을 번역한다고 가정하겠습니다.

- 입력 문장: `The cat is on the mat.`
- 타겟 문장: `Le chat est sur le tapis.`

Encoder는 입력 문장을 읽고 source 문장의 표현을 만듭니다.  
Decoder는 Encoder가 만든 표현을 참고하여 target 문장을 생성합니다.

---

# 1. Tokenizer

입력 문장은 Tokenizer를 통해 여러 개의 token으로 분할됩니다. 이후 각 token은 미리 정의된 vocabulary에 따라 token id로 변환됩니다.

실제 모델에서는 단어 단위가 아니라 subword 단위로 분할되는 경우가 많습니다. 다만 여기서는 이해를 돕기 위해 단어 단위로 단순화하겠습니다.

입력 문장:

```text
The cat is on the mat.
```

단어 단위로 단순화하면 다음과 같이 분할할 수 있습니다.

```text
["The", "cat", "is", "on", "the", "mat"]
```

그리고 vocabulary에 따라 다음과 같은 token id로 매핑된다고 가정하겠습니다.

```text
[1, 5, 2, 6, 1, 10]
```

여기서 token id는 모델이 직접 의미를 이해하는 값이 아니라, vocabulary에서 특정 token을 가리키는 index입니다. 따라서 token id는 다음 단계에서 embedding vector로 변환되어야 합니다.

---

# 2. Embedding

Token id는 단순한 숫자이기 때문에 모델이 바로 의미를 이해할 수 없습니다. 그래서 Embedding Layer를 통해 각 token id를 vector로 변환합니다.

앞선 예시에서 token id가 다음과 같다면,

```text
[1, 5, 2, 6, 1, 10]
```

Embedding Layer를 거쳐 다음과 같은 vector 표현으로 변환됩니다.

```text
[E_1, E_5, E_2, E_6, E_1, E_10]
```

Embedding Layer는 일종의 lookup table입니다. 이 lookup table은 학습 가능한 가중치 행렬로 표현할 수 있습니다.

$$
W_{emb} \in \mathbb{R}^{V \times d_{model}}
$$

여기서 각 기호의 의미는 다음과 같습니다.

- $V$: vocabulary에 정의된 token의 개수
- $d_{model}$: Transformer 내부에서 하나의 token을 표현하는 vector 차원

예를 들어 `"The"`에 해당하는 token id가 `1`이라고 가정하면, Embedding Layer는 $W_{emb}$에서 1번 row를 가져옵니다.

$$
E_1 := W_{emb}[1]
$$

즉, token id는 embedding matrix의 특정 row를 선택하기 위한 index 역할을 합니다.

---

# 3. Positional Encoding

Embedding Layer를 지난 데이터에는 token의 위치 정보가 포함되어 있지 않습니다.

예를 들어 첫 번째 `"The"`와 다섯 번째 `"the"`가 같은 token id를 가진다고 가정하면, 두 token은 같은 embedding vector $E_1$로 표현됩니다.

```text
[E_1, E_5, E_2, E_6, E_1, E_10]
```

하지만 두 token은 문장 안에서 서로 다른 위치에 있습니다. Transformer가 이 차이를 알 수 있도록 각 token embedding에 위치 정보를 더해줍니다. 이를 **Positional Encoding**이라고 합니다.

$$
Input = Embedding + PositionalEncoding
$$

Positional Encoding은 embedding vector와 같은 $d_{model}$ 차원을 가집니다. 그래야 두 vector를 더할 수 있기 때문입니다.

원 논문에서는 위치 정보를 만들기 위해 sin, cos 함수를 사용했습니다.

$$
PE_{(pos, 2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

여기서 각 기호의 의미는 다음과 같습니다.

- $pos$: token의 위치
- $i$: embedding vector 안의 차원 index
- $d_{model}$: embedding vector 차원

핵심은 각 위치마다 서로 다른 패턴의 vector를 만들어준다는 점입니다. 같은 token이라도 위치 정보가 더해지면 Transformer에 입력되는 값은 달라집니다.

예를 들어 같은 token id를 가진 두 token이라도 다음처럼 서로 다른 입력이 됩니다.

$$
E_1 + PE_{pos=1}
$$

$$
E_1 + PE_{pos=5}
$$

---

# 4. Encoder-Decoder 구조

Transformer 원 논문은 Encoder-Decoder 구조를 사용합니다.

원 논문 기준으로 Encoder는 6개의 Encoder Layer로 구성되고, Decoder는 6개의 Decoder Layer로 구성됩니다.

즉, 전체 구조는 다음과 같이 볼 수 있습니다.

```text
Encoder Layer × 6
Decoder Layer × 6
```

Encoder는 source 문장을 처리합니다.

```text
The cat is on the mat.
```

Decoder는 target 문장을 생성합니다.

```text
Le chat est sur le tapis.
```

Encoder와 Decoder의 역할은 다음과 같습니다.

| 구분 | 역할 |
|---|---|
| Encoder | 입력 문장을 읽고 source representation 생성 |
| Decoder | 이전 target token과 Encoder 출력을 참고하여 다음 token 생성 |

---

# 5. Encoder 구조와 Multi-Head Attention

입력 텍스트는 Embedding Layer와 Positional Encoding을 거쳐 $d_{model}$ 차원의 vector로 변환됩니다.

첫 번째 Encoder Layer는 입력 문장의 embedding 결과를 입력으로 받습니다. 두 번째 Encoder Layer부터는 이전 Encoder Layer의 출력값을 입력으로 받습니다.

Encoder Layer는 크게 두 개의 sub-layer로 구성됩니다.

1. Multi-Head Self-Attention
2. Feed Forward Network

각 sub-layer 뒤에는 Residual Connection과 Layer Normalization이 적용됩니다.

```text
Input
 → Multi-Head Self-Attention
 → Add & Norm
 → Feed Forward Network
 → Add & Norm
 → Output
```

---

## 5-1. Self-Attention에서 Q, K, V 생성

Encoder의 Self-Attention에서는 입력 vector $X$로부터 Query, Key, Value를 생성합니다.

$$
Q = XW^Q
$$

$$
K = XW^K
$$

$$
V = XW^V
$$

여기서 $W^Q$, $W^K$, $W^V$는 학습 가능한 weight matrix입니다.

만약 입력 $X$의 shape이 다음과 같다면,

$$
X \in \mathbb{R}^{seq\_len \times d_{model}}
$$

각 weight matrix는 보통 다음과 같은 shape을 가집니다.

$$
W^Q \in \mathbb{R}^{d_{model} \times d_k}
$$

$$
W^K \in \mathbb{R}^{d_{model} \times d_k}
$$

$$
W^V \in \mathbb{R}^{d_{model} \times d_v}
$$

따라서 Q, K, V의 shape은 다음과 같습니다.

$$
Q \in \mathbb{R}^{seq\_len \times d_k}
$$

$$
K \in \mathbb{R}^{seq\_len \times d_k}
$$

$$
V \in \mathbb{R}^{seq\_len \times d_v}
$$

Query, Key, Value는 다음과 같이 이해할 수 있습니다.

- **Query**: 현재 위치가 찾고 싶은 정보
- **Key**: 참고 대상 token의 검색용 특징
- **Value**: 실제로 전달되는 정보

다만 중요한 점은 Query, Key, Value가 처음부터 이런 의미를 가진 것은 아니라는 점입니다. 학습 초기의 $W^Q$, $W^K$, $W^V$는 랜덤하게 초기화되어 있으므로 Q, K, V에는 특별한 의미가 없습니다.

이 역할은 Attention 수식 안에서 맡는 위치와 loss를 줄이는 학습 과정에 의해 형성됩니다.

---

## 5-2. Scaled Dot-Product Attention

생성된 Q, K, V는 Scaled Dot-Product Attention에 입력됩니다.

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

먼저 Q와 K를 곱합니다.

$$
QK^T
$$

Q의 shape은 $(seq\_len, d_k)$이고, K의 shape도 $(seq\_len, d_k)$입니다. 따라서 $K^T$의 shape은 $(d_k, seq\_len)$입니다.

결과적으로 $QK^T$의 shape은 다음과 같습니다.

$$
(seq\_len, d_k) \times (d_k, seq\_len) = (seq\_len, seq\_len)
$$

이 행렬은 각 token이 다른 token을 얼마나 참고해야 하는지를 나타냅니다. 이를 **Attention Score Matrix**라고 합니다.

각 원소는 다음과 같이 표현할 수 있습니다.

$$
score_{ij} = (QK^T)_{ij} = q_i \cdot k_j
$$

여기서 $q_i$는 i번째 token의 Query vector이고, $k_j$는 j번째 token의 Key vector입니다.

즉, $score_{ij}$는 i번째 token이 j번째 token을 얼마나 참고해야 하는지를 나타내는 값이라고 볼 수 있습니다.

---

## 5-3. Query, Key, Value는 왜 이런 역할로 학습되는가?

Query와 Key의 내적 결과는 두 token 사이의 관련도 또는 참조 강도를 나타낸다고 해석할 수 있습니다. 이렇게 모든 Query와 Key의 내적을 계산한 행렬을 **Attention Score Matrix**라고 합니다.

다만 중요한 점은 Query와 Key가 처음부터 “내가 찾고 싶은 정보”와 “내가 가진 정보의 특징”을 의미하는 것은 아니라는 점입니다. 학습 초기의 $W^Q$, $W^K$, $W^V$는 랜덤하게 초기화되어 있으므로, 이때 생성되는 Query, Key, Value에는 특별한 의미가 없습니다.

하지만 학습이 진행되면 모델은 정답 token의 확률을 높이는 방향으로 loss를 줄입니다. 이 과정에서 역전파를 통해 $W^Q$, $W^K$, $W^V$가 업데이트됩니다.

만약 어떤 token을 참고하는 것이 정답 예측에 도움이 된다면, 해당 token 쌍의 Attention Score는 커지는 방향으로 학습됩니다. 반대로 정답 예측에 도움이 되지 않는 token 쌍의 Attention Score는 작아지는 방향으로 학습됩니다.

이때 Query와 Key의 역할은 Attention Score Matrix 안에서의 위치로 구분할 수 있습니다.

$$
score_{ij} = q_i \cdot k_j
$$

여기서 $q_i$는 현재 Query 위치, $k_j$는 참고 후보 Key 위치입니다. 즉 Query는 Attention Score Matrix의 row를 만들고, Key는 column 방향에서 여러 Query와 비교됩니다.

따라서 Query는 “현재 위치가 정답 예측을 위해 무엇을 참고해야 하는가”를 표현하는 방향으로 학습되고, Key는 “각 token이 다른 위치에서 참고될 때 어떤 특징으로 매칭될 수 있는가”를 표현하는 방향으로 학습됩니다.

Value는 Attention Weight에 따라 실제로 전달되는 정보를 담는 방향으로 학습됩니다. Query와 Key의 내적은 “어디를 얼마나 참고할지”를 결정하고, Value는 그렇게 선택된 위치에서 실제로 다음 layer로 전달될 정보를 담습니다.

---

## 5-4. 왜 $\sqrt{d_k}$로 나누는가?

Query와 Key의 내적값은 $d_k$개의 항을 더해서 계산됩니다.

$$
q_i \cdot k_j = \sum_{r=1}^{d_k} q_{i,r}k_{j,r}
$$

따라서 $d_k$가 커질수록 내적값의 분산이 커질 수 있습니다. 내적값이 너무 커지면 Softmax의 출력이 특정 token에 과도하게 집중될 수 있습니다. 이 경우 Softmax의 gradient가 작아져 학습이 불안정해질 수 있습니다.

이를 완화하기 위해 Transformer는 Attention Score를 $\sqrt{d_k}$로 나누어 scaling합니다.

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

만약 Query와 Key의 각 원소가 평균 0, 분산 1을 가진다고 가정하면, 내적 $q_i \cdot k_j$의 분산은 대략 $d_k$에 비례합니다. 따라서 이를 $\sqrt{d_k}$로 나누면 분산 scale이 다시 안정됩니다.

즉, $\sqrt{d_k}$로 나누는 이유는 Attention Score가 지나치게 커져 Softmax가 한쪽으로 치우치는 것을 방지하기 위해서입니다.

---

## 5-5. Softmax와 Attention Weight

Attention Score Matrix는 Softmax를 거쳐 Attention Weight로 변환됩니다.

$$
AttentionWeight = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

Softmax는 각 row에 대해 적용됩니다. 따라서 각 token은 자신이 참고할 수 있는 token들에 대해 attention weight를 갖습니다.

예를 들어 i번째 token에 대한 Attention Weight는 다음과 같습니다.

$$
[\alpha_{i1}, \alpha_{i2}, ..., \alpha_{in}]
$$

이 값들의 합은 1입니다.

$$
\sum_j \alpha_{ij} = 1
$$

Attention Weight는 “어디를 얼마나 참고할지”를 의미합니다.

마지막으로 이 weight를 Value에 곱합니다.

$$
AttentionOutput = AttentionWeight \cdot V
$$

즉, Attention의 전체 흐름은 다음과 같습니다.

```text
Q, K 내적 → 참고할 위치의 score 계산
Softmax → 참고 비율 계산
Value 가중합 → 실제 정보 전달
```

---

# 6. Multi-Head Attention

Multi-Head Attention은 하나의 Attention만 수행하는 것이 아니라, 여러 개의 Attention Head를 병렬로 수행하는 구조입니다.

주의할 점은 Multi-Head Attention이 입력 sequence를 token 단위로 나누는 것이 아니라, 각 token representation의 feature 차원을 head별로 나누어 서로 다른 Attention을 수행한다는 점입니다.

예를 들어 원 논문에서는 다음 값을 사용했습니다.

$$
d_{model}=512
$$

$$
h=8
$$

따라서 각 head의 차원은 다음과 같습니다.

$$
d_k=d_v=\frac{d_{model}}{h}=64
$$

각 head는 서로 다른 $W_i^Q$, $W_i^K$, $W_i^V$를 사용하여 Q, K, V를 생성합니다.

$$
head_i = Attention(Q_i, K_i, V_i)
$$

각 head에서 계산된 결과는 concat됩니다.

$$
Concat(head_1, ..., head_h)
$$

이후 최종 projection matrix $W^O$를 곱해 다시 $d_{model}$ 차원으로 변환합니다.

$$
MultiHead(Q,K,V)=Concat(head_1,\dots,head_h)W^O
$$

Multi-Head Attention을 사용하면 모델은 같은 문장 안에서도 서로 다른 관점의 관계를 학습할 수 있습니다.

예를 들어 다음과 같은 역할 분담이 가능합니다.

- 어떤 head는 문법적 관계 학습
- 어떤 head는 의미적 관계 학습
- 어떤 head는 멀리 떨어진 token 간 관계 학습
- 어떤 head는 특정 위치 주변의 local pattern 학습

즉, Multi-Head Attention은 여러 관점에서 token 간 관계를 동시에 학습하기 위한 구조입니다.

---

# 7. Residual Connection과 Layer Normalization

Transformer의 각 sub-layer 뒤에는 Residual Connection과 Layer Normalization이 적용됩니다.

$$
Output = LayerNorm(x + Sublayer(x))
$$

Residual Connection은 입력 $x$를 sub-layer 출력에 더해주는 구조입니다.

이 구조를 사용하면 gradient가 더 잘 흐르고, 깊은 layer를 쌓아도 학습이 안정적으로 진행됩니다.

중요한 점은 Residual Connection을 하려면 입력과 출력의 차원이 같아야 한다는 것입니다. 따라서 Transformer의 대부분의 sub-layer는 최종적으로 $d_{model}$ 차원의 representation을 출력하도록 설계되어 있습니다.

---

# 8. Feed Forward Network

Multi-Head Attention을 거친 출력은 이후 **Feed Forward Network**를 통과합니다. Transformer의 Encoder Layer는 크게 두 단계로 볼 수 있습니다.

1. Multi-Head Attention
2. Feed Forward Network

Multi-Head Attention이 token 간의 관계를 계산하는 역할이라면, Feed Forward Network는 Attention 결과로 만들어진 각 token의 representation을 한 번 더 변환하는 역할을 합니다.

즉, Attention은 다음 질문에 답합니다.

> “이 token은 문장 안의 어떤 token들을 참고해야 하는가?”

반면 Feed Forward Network는 다음 질문에 답합니다.

> “참고 정보를 반영한 이 token representation을 어떻게 더 유용한 feature로 변환할 것인가?”

---

## 8-1. Position-wise Feed Forward Network

Transformer의 FFN은 각 token 위치마다 동일하게 적용되는 작은 MLP입니다. 이를 **Position-wise Feed Forward Network**라고 합니다.

수식은 다음과 같습니다.

$$
FFN(x)=max(0, xW_1+b_1)W_2+b_2
$$

원 논문에서는 ReLU activation을 사용했습니다.

여기서 $x$는 하나의 token representation입니다. 만약 입력 sequence의 길이가 6이라면, FFN은 6개의 token vector 각각에 독립적으로 적용됩니다.

$$
X = [x_1, x_2, x_3, x_4, x_5, x_6]
$$

$$
FFN(X) = [FFN(x_1), FFN(x_2), FFN(x_3), FFN(x_4), FFN(x_5), FFN(x_6)]
$$

중요한 점은 FFN은 token 간 Attention을 수행하지 않는다는 것입니다. 즉, FFN 단계에서는 $x_1$이 $x_2$를 직접 참고하지 않습니다.

Token 간 정보 교환은 앞의 Multi-Head Attention에서 이미 수행됩니다. FFN은 그 결과로 만들어진 각 token의 vector를 독립적으로 변환합니다.

---

## 8-2. FFN의 차원 확장과 축소

FFN의 내부 차원은 보통 다음과 같이 확장되었다가 다시 축소됩니다.

$$
d_{model} \rightarrow d_{ff} \rightarrow d_{model}
$$

원 논문에서는 다음 값을 사용했습니다.

$$
d_{model}=512
$$

$$
d_{ff}=2048
$$

즉, 하나의 token vector가 먼저 더 큰 차원으로 확장됩니다.

$$
512 \rightarrow 2048
$$

그다음 ReLU 같은 비선형 함수를 통과한 뒤, 다시 원래 차원으로 축소됩니다.

$$
2048 \rightarrow 512
$$

이렇게 다시 $d_{model}$ 차원으로 돌아오는 이유는 Residual Connection을 위해서입니다. Transformer의 각 sub-layer는 입력과 출력을 더하는 residual 구조를 사용합니다.

$$
Output = LayerNorm(x + Sublayer(x))
$$

입력 $x$와 sub-layer 출력값을 더하려면 두 vector의 차원이 같아야 합니다. 따라서 FFN은 내부에서는 차원을 확장하지만, 최종 출력은 다시 $d_{model}$ 차원으로 맞춥니다.

---

## 8-3. FFN을 학습 관점에서 이해하기

FFN의 weight $W_1$, $W_2$도 loss를 줄이는 방향으로 업데이트됩니다.

Attention이 어떤 token을 참고할지 결정했다면, FFN은 그 결과 representation을 정답 예측에 더 유리한 feature로 바꾸는 방향으로 학습됩니다.

예를 들어 Attention 결과 어떤 token representation 안에 문법 정보, 의미 정보, 위치 정보가 섞여 있다고 하겠습니다. FFN은 이 혼합된 representation을 다음 layer나 최종 출력에 더 유용한 형태로 변환합니다.

따라서 FFN은 단순히 “비선형성을 추가하는 층”이라고만 설명하기보다는 다음과 같이 이해하는 것이 좋습니다.

> Attention이 token 간 정보를 섞는 단계라면, FFN은 섞인 정보를 각 token 위치에서 다시 가공하고 정제하는 단계입니다.

정리하면 FFN의 역할은 다음과 같습니다.

1. Attention 결과로 만들어진 각 token representation을 독립적으로 변환합니다.
2. token 간 정보를 새로 섞지는 않습니다.
3. $d_{model}$ 차원을 $d_{ff}$로 확장한 뒤 다시 $d_{model}$로 줄입니다.
4. 중간에 ReLU 같은 비선형 함수를 넣어 표현력을 높입니다.
5. 최종 출력 차원을 $d_{model}$로 맞춰 Residual Connection이 가능하게 합니다.
6. loss 역전파를 통해 정답 예측에 유리한 feature 변환을 학습합니다.

---

# 9. Decoder 구조

Decoder는 target 문장을 생성하는 역할을 합니다.

Decoder Layer는 크게 세 개의 sub-layer로 구성됩니다.

1. Masked Multi-Head Self-Attention
2. Cross-Attention
3. Feed Forward Network

각 sub-layer 뒤에는 Residual Connection과 Layer Normalization이 적용됩니다.

```text
Decoder Input
 → Masked Multi-Head Self-Attention
 → Add & Norm
 → Cross-Attention
 → Add & Norm
 → Feed Forward Network
 → Add & Norm
 → Output
```

---

## 9-1. Decoder 입력: 학습 시점

학습 시에는 정답 target 문장을 알고 있습니다. 따라서 정답 문장을 오른쪽으로 한 칸 shift한 sequence를 Decoder에 한 번에 넣습니다.

정답 target:

```text
[Le, chat, est, sur, le, tapis]
```

Decoder 입력:

```text
[<SOS>, Le, chat, est, sur, le]
```

Decoder가 예측해야 하는 target:

```text
[Le, chat, est, sur, le, tapis]
```

중요한 점은 Decoder 입력이 한 번에 들어가더라도 Look-Ahead Mask 때문에 각 위치는 미래 token을 볼 수 없다는 점입니다.

예를 들어 `chat`을 예측하는 위치에서는 `<SOS>`와 `Le`까지만 참고할 수 있습니다.

---

## 9-2. Decoder 입력: 추론 시점

추론 시에는 정답 target 문장을 모릅니다. 따라서 처음에는 `<SOS>`만 Decoder에 넣고, 모델이 예측한 token을 다시 입력 뒤에 붙이면서 하나씩 생성합니다.

```text
Step 1: [<SOS>] → Le 예측
Step 2: [<SOS>, Le] → chat 예측
Step 3: [<SOS>, Le, chat] → est 예측
Step 4: 반복 → Le chat est sur le tapis
```

---

## 9-3. Masked Self-Attention

Decoder의 첫 번째 sub-layer는 Masked Multi-Head Self-Attention입니다.

학습 시 Decoder 입력은 전체 sequence로 들어갑니다.

```text
[<SOS>, Le, chat, est, sur, le]
```

그러나 Look-Ahead Mask를 적용하기 때문에 각 위치는 미래 token을 볼 수 없습니다.

예를 들어 Decoder 입력이 다음과 같다고 하겠습니다.

```text
[<SOS>, Le, chat, est]
```

Look-Ahead Mask는 다음과 같은 형태를 가집니다.

| Query 위치 \ Key 위치 | `<SOS>` | `Le` | `chat` | `est` |
|---|---:|---:|---:|---:|
| `<SOS>` | 0 | $-\infty$ | $-\infty$ | $-\infty$ |
| `Le` | 0 | 0 | $-\infty$ | $-\infty$ |
| `chat` | 0 | 0 | 0 | $-\infty$ |
| `est` | 0 | 0 | 0 | 0 |

여기서 0은 참고 가능하다는 뜻이고, $-\infty$는 참고할 수 없다는 뜻입니다.

계산 순서는 다음과 같습니다.

$$
Q = X_{dec}W^Q
$$

$$
K = X_{dec}W^K
$$

$$
V = X_{dec}W^V
$$

먼저 Attention Score Matrix를 계산합니다.

$$
AttentionScoreMatrix = QK^T
$$

그다음 Mask를 더합니다.

$$
MaskedScore = AttentionScoreMatrix + Mask
$$

이후 softmax를 적용합니다.

$$
AttentionWeight = softmax\left(\frac{MaskedScore}{\sqrt{d_k}}\right)
$$

$-\infty$가 들어간 위치는 softmax 이후 attention weight가 0이 됩니다. 따라서 Decoder는 미래 token을 참고하지 못합니다.

---

## 9-4. Cross-Attention

Decoder의 두 번째 sub-layer는 Cross-Attention입니다.

Cross-Attention에서는 Decoder의 현재 상태가 Query가 되고, Encoder 출력 $H_{enc}$가 Key와 Value가 됩니다.

$$
Q_{cross} = H_{dec}W^Q
$$

$$
K_{cross} = H_{enc}W^K
$$

$$
V_{cross} = H_{enc}W^V
$$

즉, Cross-Attention에서는 다음과 같이 볼 수 있습니다.

| 역할 | 출처 |
|---|---|
| Query | Decoder hidden state |
| Key | Encoder output |
| Value | Encoder output |

Cross-Attention의 목적은 Decoder가 현재 target token을 생성하기 위해 source 문장의 어떤 token을 참고해야 하는지 계산하는 것입니다.

예를 들어 `chat`을 예측해야 하는 시점에서는 source 문장의 `cat` 정보가 중요합니다. 따라서 학습이 잘 된 모델이라면 Decoder Query는 Encoder의 `cat` Key와 높은 attention score를 가지게 됩니다.

$$
q_{\text{predict chat}} \cdot k_{\text{cat}}
$$

이 값이 커지면 `cat` 위치의 attention weight가 커지고, `cat`의 Value 정보가 Decoder에 더 많이 반영됩니다.

---

# 10. Padding Mask와 Look-Ahead Mask

Transformer에서는 주로 두 종류의 mask가 사용됩니다.

## 10-1. Padding Mask

입력 문장의 길이는 서로 다를 수 있습니다. 하지만 batch 연산을 위해서는 sequence 길이를 맞춰야 합니다. 이때 짧은 문장 뒤에 `<PAD>` token을 붙입니다.

예를 들어 다음과 같습니다.

```text
The cat is on the mat <PAD> <PAD>
```

Padding Mask는 `<PAD>` 위치가 attention 계산에 반영되지 않도록 막는 역할을 합니다.

즉, `<PAD>` token에 attention weight가 분배되지 않도록 해당 위치의 score를 $-\infty$로 만듭니다.

Padding Mask는 다음 위치에서 사용될 수 있습니다.

- Encoder Self-Attention
- Decoder Masked Self-Attention
- Decoder Cross-Attention

## 10-2. Look-Ahead Mask

Look-Ahead Mask는 Decoder의 Masked Self-Attention에서 사용됩니다.

Decoder는 target 문장을 생성할 때 미래 token을 보면 안 됩니다. 따라서 현재 위치보다 오른쪽에 있는 token을 참고하지 못하도록 막습니다.

예를 들어 Decoder 입력이 다음과 같다면,

```text
[<SOS>, Le, chat, est]
```

`Le` 위치는 `<SOS>`와 `Le`까지만 볼 수 있고, `chat`, `est`는 볼 수 없습니다.

Look-Ahead Mask는 올바른 인과관계를 유지하기 위한 mask입니다.

---

# 11. 최종 출력

Decoder의 마지막 layer를 통과한 output은 Linear Layer와 Softmax를 거칩니다.

Linear Layer는 Decoder hidden state를 vocabulary 크기의 logit vector로 변환합니다.

$$
Logits = H_{dec}W_{vocab} + b
$$

그다음 Softmax를 적용하여 각 token에 대한 확률을 계산합니다.

$$
P(y_t|y_{<t},x)=softmax(Logits)
$$

예를 들어 `chat`을 예측해야 하는 위치라면, 모델은 vocabulary 전체에 대한 확률을 계산합니다.

| token | 확률 예시 |
|---|---:|
| Le | 0.01 |
| chat | 0.82 |
| est | 0.05 |
| tapis | 0.02 |

학습 시에는 정답 token의 확률을 높이는 방향으로 loss를 줄입니다.

$$
Loss = -\log P(y_t)
$$

추론 시에는 확률이 가장 높은 token을 선택하거나, beam search 같은 방법을 사용해 다음 token을 선택합니다.

---

# 12. 전체 정리

Transformer의 전체 흐름은 다음과 같습니다.

1. 입력 문장을 token으로 나눕니다.
2. token을 token id로 변환합니다.
3. token id를 embedding vector로 변환합니다.
4. embedding vector에 positional encoding을 더합니다.
5. Encoder는 source 문장을 처리하여 $H_{enc}$를 만듭니다.
6. Decoder는 shifted target을 입력받습니다.
7. Decoder의 Masked Self-Attention은 미래 token을 보지 못하도록 막습니다.
8. Decoder의 Cross-Attention은 Encoder 출력 $H_{enc}$를 참고합니다.
9. Feed Forward Network는 각 token representation을 독립적으로 변환합니다.
10. 마지막 Linear Layer와 Softmax를 통해 다음 token 확률을 계산합니다.
11. 학습 시에는 정답 token의 확률을 높이도록 loss를 줄입니다.
12. 추론 시에는 `<SOS>`부터 시작해 token을 하나씩 생성합니다.

Transformer를 한 문장으로 정리하면 다음과 같습니다.

> Attention을 통해 token 간 관계를 계산하고, FFN을 통해 각 token representation을 변환하면서, 입력 문장을 출력 문장으로 변환하는 Encoder-Decoder 구조입니다.
