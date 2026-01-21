# - 자치구별 1인가구가 살기 적합한 곳을 탐색하고자 함- 

# 변수 = - 주거비용, 범죄발생, 범죄검거, 교통, 인프라, 의료시설, 1인가구 수
# 1인가구 수, 범죄발생 건수, 범죄검거 건수, 인프라 수, 지하철 수, 의료시설 수

import pandas as pd
import numpy as np  # 데이터 처리용

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 시각화용

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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



file_path = "서울시_1인가구수4.xlsx"
df = pd.read_excel(file_path)

data = df.iloc[4:, :]  # 상단 설명 행 제외


data = data.rename(columns={
    data.columns[1]:  "1인가구수",
    data.columns[10]: "평균", # ✔️ 평 당 평균 (월세)
    data.columns[11]: "범죄_발생",
    data.columns[12]: "범죄_검거",
    data.columns[15]: "지하철",
    data.columns[16]: "대규모점포수",
    data.columns[19]: "총합"   # ✔️ 총합(의료) 추가

})

data = data.set_index(data.columns[0])  #✔️ A열 = 자치구명을 index로 



selected = data[
    ["1인가구수", "평균", "범죄_발생",
     "범죄_검거", "지하철", "대규모점포수", "총합"]
].apply(pd.to_numeric, errors="coerce")   # 결측 제거

corr = selected.corr() # 상관계수

selected_clean = selected.dropna() # ✔️상관원

label_map = {
    "대규모점포수": "인프라",
    "평균": "주거비용",
    "지하철": "교통",  # ✔️검거율 삭제 / 그래프 표시명 변경
    "총합": "의료시설", # ✔️의료시설 
}

corr_plot = corr.rename(index=label_map, columns=label_map) 



features = [  # PCA 용 변수
    "평균",
    "범죄_발생",
    "범죄_검거",
    "지하철",
    "대규모점포수",
    "총합"
]

X = selected_clean[features]  

X["평균"] *= -1  # ✔️ 낮을수록 좋음  # (값이 클수록 좋은게 기본)
X["범죄_발생"] *= -1



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




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 표준화


pca = PCA()
X_pca = pca.fit_transform(X_scaled) # PCA 실행

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)


explained_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained))],
    "설명분산비율": explained,
    "누적설명분산": cumulative
})

print("\n[설명분산/누적설명분산]")
print(explained_df)

loadings = pd.DataFrame(  # 적재값 로딩 (성분분석표)
    pca.components_.T,
    index=features,
    columns=[f"PC{i+1}" for i in range(len(features))]
)

print("\n[성분분석표(loadings)]")
print(loadings)



plt.figure(figsize=(7, 4))  # ✔️ 스크리 스콧 (pc 개수 판단)
plt.plot(range(1, len(explained) + 1), explained, marker="o")
plt.xticks(range(1, len(explained) + 1))
plt.xlabel("PC")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
plt.xticks(range(1, len(cumulative) + 1))
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Cumulative Explained Variance")
plt.axhline(0.7, linestyle="--")
plt.axhline(0.8, linestyle="--")
plt.tight_layout()
plt.show()


n_pc = 4  #  ✔️ 보통 2~4 사이에서 결정  # pca 점수 만들기 + 1인가구 붙이기

scores = pd.DataFrame(
    X_pca[:, :n_pc],
    index=selected_clean.index,
    columns=[f"PC{i+1}" for i in range(n_pc)]
)

scores["1인가구수"] = selected_clean["1인가구수"]

print("\n[PCA 점수(scores) + 1인가구수]")
print(scores.head())



if n_pc >= 2:  #  ✔️ pc1-pc2 산점도
    plt.figure(figsize=(9, 8))
    plt.scatter(scores["PC1"], scores["PC2"], alpha=0.7)

    dx = -0.25
    dy = -0.05  # ✔️ 글씨가 점과 너무 가까울때 dx/dy ↑ <-> 전체적으로 밀릴때

    for gu in scores.index: 
        plt.text(
            scores.loc[gu, "PC1" + dx],
            scores.loc[gu, "PC2" + dy],
            gu,
            fontsize = 7, #  ✔️글씨 겹침이 심할때 fontsize ↑
            ha = "left",
            va = "top" #  ✔️ 오프셋 크기
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("자치구 분포: PC1 vs PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


weights = explained[:n_pc]  # '자취 적합 지수' 만들기
weights = weights / weights.sum()

scores["자취적합지수"] = 0
for i in range(n_pc):
    scores["자취적합지수"] += weights[i] * scores[f"PC{i+1}"]

# 지수 상위/하위 출력 (인덱스가 자치구명이면 더 보기 좋음)
print("\n[자취적합지수 상위 10]")
print(scores.sort_values("자취적합지수", ascending=False).head(10))

print("\n[자취적합지수 하위 10]")
print(scores.sort_values("자취적합지수", ascending=True).head(10))



corr_index = scores["자취적합지수"].corr(scores["1인가구수"])  #  ✔️ 자취적합지수 vs 1인가구수 상관확인
print(f"\n[자취적합지수와 1인가구수 상관계수] {corr_index:.4f}")




