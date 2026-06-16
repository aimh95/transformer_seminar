# Transformer Learning Style Explainer

## 목적

이 자료는 Transformer를 개념 암기가 아니라 **tensor flow, shape, mask, loss/gradient 관점**에서
이해하는 방식을 설명하기 위해 제작되었다.

"Attention은 관계를 본다"는 설명에 만족하지 않고,
Q/K/V가 어떤 행렬 연산에서 나오는지, Mask가 정확히 어디에 적용되는지,
FFN과 Attention이 어떻게 다른지를 실제 연산 그래프로 납득하는 사람을 위한 자료다.

---

## 참고 자료

- **Attention Is All You Need**, Vaswani et al., 2017
- arXiv:1706.03762

본 자료의 모든 도식은 위 논문의 Figure 1 (Transformer Architecture)과
Figure 2 (Scaled Dot-Product Attention / Multi-Head Attention)를 참고하여
HTML/SVG로 재도식화한 것이다. 원본 이미지는 포함되지 않았다.

---

## 실행 방법

```bash
# 브라우저에서 바로 열기
open transformer_learning_style_explainer/index.html    # macOS
xdg-open transformer_learning_style_explainer/index.html # Linux
start transformer_learning_style_explainer/index.html   # Windows
```

외부 서버 불필요. `index.html`을 브라우저에서 열면 된다.
단, `assets/` 폴더가 `index.html`과 같은 디렉토리에 있어야 SVG가 정상 표시된다.

---

## 산출물

```
transformer_learning_style_explainer/
  index.html                              # 메인 학습 자료 (standalone HTML)
  assets/
    transformer_architecture_redraw.svg  # Encoder-Decoder 전체 구조
    attention_qkv_matrix_redraw.svg      # Q/K/V 행렬 연산 및 Score Matrix
    mask_matrix_redraw.svg               # Look-Ahead Mask 행렬 및 계산 순서
    ffn_vs_attention_redraw.svg          # FFN vs Attention 비교
    engineering_mindset_flow.svg         # 3단계 이해 깊이 전략
  preview.png                            # (생성 시도됨, 아래 참조)
  README.md                              # 이 파일
```

---

## HTML 구성 (8개 섹션)

| 섹션 | 내용 |
|------|------|
| 1 | 일반적인 이해 vs 연산 그래프 기반 이해 비교 |
| 2 | Transformer 전체 구조 (Encoder-Decoder 연산 흐름) |
| 3 | Q/K/V 역할이 연산 구조에서 어떻게 성립하는가 |
| 4 | Look-Ahead Mask 위치 및 계산 순서 (인터랙티브 표 포함) |
| 5 | FFN vs Attention 차이 (token mixing vs per-token 변환) |
| 6 | 이 이해 방식의 실무적 장점 6가지 |
| 7 | 이 이해 방식의 실무적 단점 5가지 + 대응 전략 |
| 8 | 3단계 이해 깊이 균형 전략 |

---

## 인터랙션

- **[개념 설명 보기] / [연산 그래프 보기] / [실무 관점 보기]** 버튼으로 관련 섹션 하이라이트
- **Mask Matrix 인터랙티브 표**: −∞ 셀에 hover 시 "왜 막히는지" 툴팁 표시

---

## SVG 도식 재도식화 방법

논문 PDF에서 이미지를 직접 추출하지 않았다.
논문 Figure 1, 2의 구조를 참고하여 SVG XML로 새롭게 그렸다.

논문 PDF에서 이미지를 직접 추출하려면 아래 방법을 사용할 수 있다:

```python
# PyMuPDF를 사용한 PDF 이미지 추출
import fitz  # pip install pymupdf

doc = fitz.open("attention_is_all_you_need.pdf")
for page_num in [2, 3]:  # Figure 1 (p.3), Figure 2 (p.4)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(3, 3)  # 3x 해상도
    clip = page.get_pixmap(matrix=mat)
    clip.save(f"figure_page{page_num+1}.png")
```

단, 논문 이미지는 저작권이 있으므로 직접 게시 시 주의가 필요하다.
학습 목적의 재도식화(redraw)는 허용 범위 내로 판단된다.

---

## preview.png 생성

Playwright/Puppeteer 또는 별도 headless 브라우저가 설치된 환경에서:

```bash
# Playwright (Python)
pip install playwright
playwright install chromium
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    page.goto('file://$(pwd)/transformer_learning_style_explainer/index.html')
    page.screenshot(path='transformer_learning_style_explainer/preview.png')
    browser.close()
"
```

```bash
# Puppeteer (Node.js)
npx puppeteer-screenshot \
  --url "file://$(pwd)/transformer_learning_style_explainer/index.html" \
  --output "transformer_learning_style_explainer/preview.png" \
  --width 1280 --height 900
```
