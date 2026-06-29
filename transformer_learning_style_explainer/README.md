# Attention 수식의 역사적 배경

## 목적

Transformer의 attention 수식 `Softmax(QKᵀ/√d_k)V`가 어떤 연구 흐름을 거쳐
지금의 형태로 수렴했는지를 시대순으로 설명하는 자료다.

"Attention은 관계를 본다"가 아니라, 왜 score를 dot product로 계산하는지,
왜 softmax가 들어갔는지, Key와 Value가 왜 분리되어 있는지, √d_k가 왜 마지막에
붙는지를 Seq2Seq → Bahdanau → Luong → Memory Networks → Transformer 순서로 따라간다.

> 이 폴더는 원래 Transformer를 tensor flow/shape/mask 관점에서 설명하는
> 16개 섹션짜리 메커니즘 walkthrough였다. 그 내용이 `transformer_seminar_html/`과
> 중복되어, 메커니즘 섹션은 모두 제거하고 **역사적 배경 5개 섹션만 남겼다.**

---

## 실행 방법

```bash
open transformer_learning_style_explainer/index.html    # macOS
xdg-open transformer_learning_style_explainer/index.html # Linux
start transformer_learning_style_explainer/index.html   # Windows
```

외부 서버 불필요. `index.html`을 브라우저에서 열면 된다.
단, `assets/katex/` 폴더가 `index.html`과 같은 디렉토리에 있어야 수식이 정상 렌더링된다
(KaTeX를 CDN이 아니라 로컬에 저장해 두었기 때문에 완전 오프라인에서도 동작한다).

---

## 산출물

```
transformer_learning_style_explainer/
  index.html                 # 메인 자료 (standalone HTML, KaTeX로 수식 렌더링)
  assets/
    katex/                   # KaTeX 로컬 저장본 (css/js/woff2 폰트)
    *.svg                     # 이전 메커니즘 walkthrough에서 쓰던 도식 (현재 index.html에서는 미사용)
  preview.png                 # 이전 버전 스크린샷 (현재 내용과 다름, 참고용)
  README.md                   # 이 파일
```

`assets/*.svg`는 더 이상 `index.html`에서 참조되지 않는다. 필요 없으면 삭제해도 된다.

---

## 구성 (5개 섹션)

| 섹션 | 내용 |
|------|------|
| 1 | Seq2Seq의 고정 길이 vector 병목 (Cho et al. 2014, Sutskever et al. 2014) |
| 2 | Bahdanau Attention — score → softmax → weighted sum, soft alignment (2014) |
| 3 | Luong Attention — dot-product score의 등장, QKᵀ의 직접적 전조 (2015) |
| 4 | Memory Networks / Key-Value Memory Networks — Key·Value 분리 관점 (2014~2016) |
| 5 | Transformer의 종합 — QKᵀ·Softmax·V·√d_k 각각이 왜 지금 형태인지 정리 |

---

## 수식 렌더링 (KaTeX)

수식은 plain-text 근사(`a_ij`, `Sigma_k` 같은 표기)가 아니라 **KaTeX로 실제 LaTeX 타이프셋**되어 있다.

- `assets/katex/katex.min.css`, `katex.min.js`, `auto-render.min.js`, `fonts/*.woff2`를
  npm 패키지 `katex@0.17.0`에서 받아 로컬에 저장했다 (CDN 의존 없음, 폰트는 woff2만 포함).
- `index.html` 하단 스크립트가 `$...$`(inline), `\[...\]`(display) 구문을 스캔해 자동으로 렌더링한다.
- Q/K/V는 `\textcolor{...}{}`로 페이지의 Q(빨강)/K(파랑)/V(초록) 색상 체계와 맞춰 색칠되어 있다.

KaTeX를 업그레이드하려면:

```bash
npm pack katex@<version>
tar xzf katex-<version>.tgz
cp package/dist/katex.min.css package/dist/katex.min.js package/dist/contrib/auto-render.min.js \
   transformer_learning_style_explainer/assets/katex/
cp package/dist/fonts/*.woff2 transformer_learning_style_explainer/assets/katex/fonts/
```

---

## 참고 논문

- Vaswani, A. et al. (2017). *Attention Is All You Need.* arXiv:1706.03762.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* arXiv:1406.1078.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS.
- Bahdanau, D., Cho, K., & Bengio, Y. (2014/2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv:1409.0473.
- Luong, M., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation.* arXiv:1508.04025.
- Weston, J., Chopra, S., & Bordes, A. (2014). *Memory Networks.* arXiv:1410.3916.
- Graves, A., Wayne, G., & Danihelka, I. (2014). *Neural Turing Machines.* arXiv:1410.5401.
- Miller, A. et al. (2016). *Key-Value Memory Networks for Directly Reading Documents.* arXiv:1606.03126.
