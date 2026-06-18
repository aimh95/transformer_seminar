# Transformer 전체 흐름 애니메이션 발표자 스크립트

## 발표 목적

이 애니메이션은 Section 0 "전체 흐름 한눈에 보기"의 Encoder-Decoder 그림을 고정한 채,
그 위에서 X(입력)와 H(Encoder 출력), Target 문장이 실제로 어떻게 움직이는지를 보여주는 화면이다.
화면에는 도식과 강조 표시, 움직이는 점(dot)만 있고 긴 설명문은 없다.
이 문서는 발표자가 Step을 넘기면서 그대로 말할 수 있도록 작성한 스크립트다.

범위는 **추론(Inference) 한 step**으로 한정한다. 학습/Loss/Backpropagation은 다루지 않는다.
목표는 "Encoder는 문장 전체를 한 번에 읽어 H를 만들고, Decoder는 그 H를 계속 참고하면서
지금까지 만든 Target 문장 뒤에 token을 하나씩 추가해 나간다"는 추론 흐름을 직관적으로 이해시키는 것이다.

움직이는 점(X, H, Target, 예측 token)에 마우스를 올리면 예시 문장과 예시 vector를 보여주는
tooltip이 뜬다. 청중이 궁금해하면 발표자가 직접 hover해서 보여줄 수 있다.

## 발표 전 안내 멘트

```text
이번에는 Transformer 내부의 세세한 계산이 아니라,
"입력 X가 어떻게 H가 되고, 추론할 때 Decoder가 그 H를 어떻게 사용하는지"를
큰 흐름으로 보겠습니다.

화면의 Encoder-Decoder 그림은 고정되어 있고,
그 위에서 점이 움직이는 것으로 X, H, Target 문장이 어떻게 흐르는지 보여드리겠습니다.

오늘 보는 것은 학습이 아니라 추론(inference) 상황입니다.
즉 이미 학습이 끝난 모델이 번역 문장을 한 token씩 만들어내는 과정입니다.
```

---

## Step 1. Source 문장이 Encoder에 입력된다

### 화면에서 볼 것

```text
Source 문장 박스가 강조되고, 파란 점(X)이 그 위치에 나타난다
```

### 말할 내용

```text
번역할 원문, 예를 들어 "The cat is on the mat."이 Encoder 맨 아래로 들어옵니다.

이 문장은 아직 숫자가 아니라 사람이 읽는 문장 그대로입니다.
이 파란 점이 지금부터 "X"라고 부를, Encoder 안에서 움직이는 입력을 나타냅니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
이 X가 Encoder 내부의 여러 층을 통과하면서 점점 다른 표현으로 바뀝니다.
```

---

## Step 2. X가 Encoder Layer × 6을 통과하며 표현이 갱신된다

### 화면에서 볼 것

```text
Embedding → Positional Encoding → Encoder Layer × 6 박스가 차례로 강조되고,
파란 점이 그 경로를 따라 아래로 이동한다
```

### 말할 내용

```text
X는 Embedding과 Positional Encoding을 거쳐 숫자 vector가 되고,
그 다음 Encoder Layer를 6번 통과합니다.

각 층에서는 Multi-Head Self-Attention과 FFN을 거치면서
"이 단어가 문장 속 다른 단어들과 어떤 관계인지"가 점점 더 잘 반영된 vector로 바뀝니다.

중요한 건, 6개 층을 통과하는 동안 X는 같은 자리(같은 token 위치)에 머물러 있고
그 안의 내용만 점점 더 풍부해진다는 점입니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
6개 층을 모두 통과하면, Encoder는 이 문장 전체를 압축한 결과물을 내놓습니다.
```

---

## Step 3. Encoder가 H_enc를 만든다

### 화면에서 볼 것

```text
H_enc 박스가 강조되고, 점의 색이 파란색(X)에서 초록색(H)으로 바뀐다
```

### 말할 내용

```text
Encoder를 다 통과한 결과를 H_enc, 줄여서 H라고 부릅니다.

H는 입력 문장 전체의 의미와 단어 간 관계가 압축되어 담긴 표현입니다.
중요한 점은, Encoder는 이 H를 "한 번만" 계산한다는 것입니다.

Decoder가 token을 몇 개를 만들든, 이 H는 다시 계산하지 않고 그대로 재사용됩니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
이제 이 H를 들고, Decoder 쪽에서 무슨 일이 일어나는지 보겠습니다.
```

---

## Step 4. Decoder가 지금까지 생성된 Target 문장을 입력받는다

### 화면에서 볼 것

```text
Target 문장(shifted right) 박스가 강조되고, 보라색 점(Target)이 나타난다.
H를 나타내는 초록 점은 Encoder 쪽에 그대로 남아 있다
```

### 말할 내용

```text
이제 Decoder 쪽을 보겠습니다.

추론을 시작할 때 Target에는 <SOS>라는 시작 기호 하나만 있습니다.
이후 한 token이 만들어질 때마다, 그 token이 Target 문장 뒤에 계속 추가됩니다.

화면의 보라색 점은 "지금까지 만들어진 Target 문장"을 나타냅니다.
Encoder의 H(초록 점)는 이미 계산이 끝나 그 자리에 그대로 대기하고 있다는 것도 함께 봐주세요.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
이 Target 문장이 Decoder 내부에서 가장 먼저 거치는 단계가 Masked Self-Attention입니다.
```

---

## Step 5. Target이 Masked Self-Attention을 통과한다

### 화면에서 볼 것

```text
Embedding → Positional Encoding → Masked Multi-Head Self-Attention 박스가 강조되고,
보라색 점이 그 경로를 따라 이동한다
```

### 말할 내용

```text
Target 문장도 Encoder와 마찬가지로 Embedding과 Positional Encoding을 거칩니다.

그 다음 Masked Self-Attention을 통과하는데,
여기서 "Masked"가 중요합니다. Decoder는 아직 만들어지지 않은 미래 token을 보면 안 됩니다.

예를 들어 "Le chat"까지 만들었다면, 다음 token을 예측할 때
"chat" 이후에 올 단어를 미리 참고하지 못하도록 막아놓습니다.

추론 시점에는 사실 미래 token이 아직 존재하지도 않기 때문에,
이 마스크는 학습 때 병렬로 계산하면서도 같은 조건을 맞추기 위한 장치라고 이해하면 됩니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
이제 Target 정보와 Encoder의 H가 만나는 단계, Cross-Attention으로 가보겠습니다.
```

---

## Step 6. Cross-Attention에서 H와 Target이 만난다

### 화면에서 볼 것

```text
H_enc → Cross-Attention으로 가는 점선 화살표("K, V ← H_enc")와 Cross-Attention 박스가 강조된다.
초록 점(H)은 Cross-Attention 쪽으로, 보라색 점(Target)도 같은 위치로 이동해 겹친다
```

### 말할 내용

```text
이 단계가 Encoder와 Decoder가 실제로 연결되는 지점입니다.

Cross-Attention에서는 Query는 Decoder(Target) 쪽에서 오고,
Key와 Value는 Encoder가 만든 H에서 옵니다.

쉽게 말하면, "지금까지 만든 Target 문장 입장에서,
원문 H의 어떤 부분을 더 참고해야 다음 단어를 잘 만들 수 있을지"를 계산하는 단계입니다.

화면의 점선 화살표가 "K, V ← H_enc"라고 되어 있는 이유도 이것입니다.
이 H는 Decoder Layer 6개 모두가 공유해서 사용합니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
H를 참고해서 얻은 정보는 FFN을 거쳐 다음 token의 확률로 이어집니다.
```

---

## Step 7. FFN → Linear → Softmax로 다음 token 확률이 계산된다

### 화면에서 볼 것

```text
FFN → Linear → Softmax → "번역 token 확률" 박스가 차례로 강조되고,
점의 색이 보라색에서 주황색(예측 token)으로 바뀐다
```

### 말할 내용

```text
Cross-Attention까지 거친 표현은 FFN을 한 번 더 통과하고,
Linear 층에서 vocabulary 크기만큼의 점수(logits)로 바뀝니다.

그다음 Softmax를 거치면 이 점수들이 합이 1인 확률로 바뀝니다.

이 확률에서 가장 높은 token, 또는 확률에 따라 sampling된 token이
이번 step에서 Decoder가 만들어낸 다음 token입니다.
```

### 다음 Step으로 넘어가는 연결 멘트

```text
이렇게 만들어진 token은 그냥 출력되고 끝나는 게 아니라, 다시 Decoder 입력으로 돌아갑니다.
```

---

## Step 8. 예측된 token이 Target 문장 뒤에 추가되어 다시 입력된다 (자기회귀)

### 화면에서 볼 것

```text
"번역 token 확률" 박스에서 Target 문장 박스로 가는 주황색 점선 화살표(loop)가 강조되고,
점이 그 경로를 따라 Target 박스로 돌아온 뒤 다시 보라색으로 바뀐다
```

### 말할 내용

```text
방금 만든 token은 Target 문장 맨 뒤에 추가됩니다.

예를 들어 <SOS> 하나로 시작했다면, 이제 <SOS>, Le가 되고,
다음 step에서는 <SOS>, Le, chat이 되는 식으로 한 token씩 길어집니다.

그리고 Decoder는 이렇게 한 token이 늘어난 Target 문장을 가지고
Step 4부터의 과정을 다시 반복합니다.

이게 바로 자기회귀(autoregressive)입니다.
Encoder의 H는 한 번 계산된 뒤로 계속 재사용되지만,
Decoder는 token이 하나씩 늘어날 때마다 이 과정을 반복해서
문장이 끝난다는 신호(예: <EOS>)가 나올 때까지 번역 문장을 완성해 나갑니다.
```

### 마무리 멘트

```text
정리하면, Encoder는 입력 문장 전체를 한 번에 읽어 H라는 압축된 표현을 만들고,
Decoder는 그 H를 계속 참고하면서 지금까지 만든 Target 문장 뒤에 token을 하나씩 추가합니다.

X가 H로 바뀌는 것은 한 번만 일어나는 일이고,
Target이 한 token씩 길어지면서 Cross-Attention으로 H를 다시 참고하는 과정은
번역 문장이 끝날 때까지 반복됩니다.
```
