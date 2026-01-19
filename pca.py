import pandas as pd
import numpy as np  # 데이터 처리용

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 시각화용



font_candidates = [
    "NanumGothic",
    "Nanum Barun Gothic",
    "Malgun Gothic",  # Windows 폰트
    "AppleGothic"     # macOS 폰트
]

available_fonts = [f.name for f in fm.fontManager.ttflist]

for font in font_candidates:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        break

plt.rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지



file_path = "서울시_1인가구수3.xlsx"    # ✔️파일명 수정
df = pd.read_excel(file_path)

data = df.iloc[4:, :]  # 상단 설명 행 제외



data = data.rename(columns={
    data.columns[1]:  "1인가구수",
    data.columns[16]: "보증금_평균",
    data.columns[19]: "범죄_발생",
    data.columns[20]: "범죄_검거",
    data.columns[23]: "지하철",
    data.columns[24]: "대규모점포수",
    data.columns[27]: "총합"   # ✔️ 총합(의료) 추가

})



selected = data[
    ["1인가구수", "보증금_평균", "범죄_발생",
     "범죄_검거", "지하철", "대규모점포수", "총합"]
].apply(pd.to_numeric, errors="coerce")

corr = selected.corr()




label_map = {
    "대규모점포수": "인프라",
    "보증금_평균": "주거비용",
    "지하철": "교통",  # ✔️검거율 삭제 / 그래프 표시명 변경
    "총합": "의료시설", # ✔️의료시설 
}

corr_plot = corr.rename(index=label_map, columns=label_map)




plt.figure(figsize=(9, 8))
ax = plt.gca()

n = len(corr_plot.columns)

for i in range(n):
    for j in range(i + 1):
        value = corr_plot.iloc[i, j]

        ax.scatter(
            i, j,
            s=abs(value) * 2300,  # ✔️ 2500 -> 2300
            c=value,
            cmap="RdBu_r",
            vmin=-1, vmax=1
        )

        ax.text(
            i, j,
            f"{value:.2f}",
            ha="center", va="center",
            fontsize=9
        )




ax.set_xticks(range(n))
ax.set_yticks(range(n))

ax.set_xticklabels(corr_plot.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_plot.columns)




ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

ax.grid(
    which="minor",
    color="lightgray",
    linestyle="-",
    linewidth=0.8
)

ax.tick_params(which="minor", bottom=False, left=False)




plt.title("서울시 1인 가구 관련 지표 간 상관관계", fontsize=14)

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
sm.set_array([])

plt.colorbar(
    sm,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="상관계수"
)

plt.tight_layout()
plt.show()
