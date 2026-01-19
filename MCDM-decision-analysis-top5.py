"""
    [ MCDM 엔트로피 가중치 분석 프로그램 ]
목적 : 서울시 자치구의 살기 좋은 지역 Top 5를 객관적으로 선정
방법 : 엔트로피 방법을 이용한 다기준 의사결정(MCDM) 분석
작성자 : 이유나
작성일: 2025-01-19

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# matplotlib에서 한글이 깨지지 않도록 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕 폰트
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# 1. 데이터 로드 함수
def load_data(file_path):
    """
    엑셀 파일을 읽어서 데이터프레임으로 변환하는 함수
    
    Parameters (입력값):
    - file_path: 엑셀 파일의 경로 (예: 'data.xlsx')
    
    Returns (반환값):
    - df: 읽어온 데이터가 담긴 DataFrame
    """

    # openpyxl 엔진을 사용하여 엑셀 파일 읽기
    df = pd.read_excel(file_path)
    return df


# 2. 엔트로피 가중치 계산

def calculate_entropy_weights(data, criteria):
    """
    엔트로피 방법으로 각 평가 기준의 가중치를 자동 계산
    
    - 데이터의 변동성이 클수록 → 중요한 지표 → 높은 가중치
    - 데이터의 변동성이 작을수록 → 낮은 가중치
    
    Parameters(입력값) :
    - data: 정규화된 데이터 (numpy array, 0~1 사이 값)
    - criteria: 평가 기준 리스트 (예: ['평당 월세 평균', '지하철', ...])
    
    Returns(반환값) :
    - weights: 각 기준의 가중치 (합이 1.0)
    
    계산 과정:
    1. 엔트로피 계산: E_j = -k * Σ(p_ij * ln(p_ij))
    2. 다양성 지수: D_j = 1 - E_j
    3. 가중치: w_j = D_j / Σ(D_j)
    """
    m, n = data.shape   # m: 자치구 개수, n: 평가 지표 개수
    k = 1 / np.log(m)   # 엔트로피 계산을 위한 상수
    
    weights = []    # 각 기준의 가중치를 저장할 리스트

    # 각 평가 기준(열)마다 가중치 계산
    for j in range(n):
        
        # 해당 기준의 모든 값들 추출(한 열의 모든 데이터)
        column = data[:, j]

        # ln(0)값이 있으면 에러가 발생하므로 작은 값으로 대체
        column = np.where(column == 0, 1e-10, column)
        
        # 엔트로피 계산 공식
        # - 데이터의 변동성이 클수록 → 중요한 지표 → 높은 가중치
        # - 데이터의 변동성이 작을수록 → 낮은 가중치
        entropy = -k * np.sum(column * np.log(column))
        
        # 다양성 지수 계산(1 - 엔트로피)
        # dicersity 크다 → 데이터 변동성이 큼
        diversity = 1 - entropy
        weights.append(diversity)
    
    # 가중치 정규화(모든 가중치의 합이 1.0이 되도록)
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    return weights

# 3. 데이터 정규화
def normalize_data(df, criteria, is_positive):
    """
    각 평가 기준의 데이터를 0~1 범위로 정규화
    
    정규화가 필요한 이유:
    - 각 지표는 단위와 범위가 다름 (예: 월세는 만원, 지하철은 개수)
    - 정규화를 통해 모든 지표를 동일한 0~1 스케일로 변환
    
    Parameters(입력값) :
    - df: 원본 데이터프레임
    - criteria: 평가 기준 리스트
    - is_positive: 각 기준이 긍정적(True)인지 부정적(False)인지
    
    Returns(반환값) :
    - normalized_df: 정규화된 데이터프레임
    - normalized_array: 정규화된 numpy 배열 (계산용)
    
    정규화 공식 :
    - 긍정적 지표 (높을수록 좋음): (X - min) / (max - min)
    - 부정적 지표 (낮을수록 좋음): 1 - (X - min) / (max - min)
    """
    # 원본 데이터를 복사
    normalized_df = df.copy()
    
    # 각 평가 기준마다 정규화 수행
    for criterion in criteria:
        # 해당 기준의 모든 값 추출
        values = df[criterion].values
        # 최댓값과 최솟값 계산
        max_val = np.max(values)
        min_val = np.min(values)
        range_val = max_val - min_val
        
        # 범위가 0이면(모든 값이 같으면) 0.5로 설정
        if range_val == 0:
            normalized = np.full(len(values), 0.5)
        else:
            # Min-Max 정규화 공식 적용
            normalized = (values - min_val) / range_val
            
            # 부정적 지표인 경우 값을 반전 (예: 월세가 낮을수록 좋으므로 반전 필요)
            if not is_positive[criterion]:
                normalized = 1 - normalized
        # 정규화된 값으로 업데이트
        normalized_df[criterion] = normalized
    
    # 계산에 사용할 numpy 배열로 변환
    normalized_array = normalized_df[criteria].values
    return normalized_df, normalized_array

# 4. MCDM 분석 수행
def perform_mcdm_analysis(df, district_col, criteria, is_positive):
    """
    MCDM 엔트로피 가중치 분석의 전체 과정을 수행하는 메인 분석 함수
    
    분석 흐름:
    1단계: 데이터 정규화(0~1 범위로 변환)
    2단계: 엔트로피 방법으로 가중치 자동 계산
    3단계: 가중 합산으로 종합 점수 계산
    4단계: 순위 매기기
    
    Parameters (입력값):
    - df: 원본 데이터프레임
    - district_col: 지역명이 담긴 컬럼 이름
    - criteria: 평가 기준 리스트
    - is_positive: 각 기준의 긍정/부정 여부
    
    Returns (반환값):
    - results_df: 순위와 점수가 포함된 최종 결과 데이터프레임
    - weights: 각 기준의 가중치 배열
    """
    # 1. 데이터 정규화
    normalized_df, normalized_array = normalize_data(df, criteria, is_positive)
    
    # 2. 엔트로피 가중치 계산
    weights = calculate_entropy_weights(normalized_array, criteria)
    
    # 3. 가중 합산으로 종합 점수 계산
    scores = np.dot(normalized_array, weights)
    
    # 4. 결과 데이터프레임 생성
    results_df = df.copy()

    # 0-100 스케일로 변환
    results_df['종합점수'] = scores * 100  

    # 종합점수를 기준으로 순위 매기기 (높은 점수 = 높은 순위)
    results_df['순위'] = results_df['종합점수'].rank(ascending=False, method='min').astype(int) 
    
    weight_df = pd.DataFrame({
        '평가기준': criteria,
        '가중치': weights,
        '가중치(%)': weights * 100
    })
    print(f'{weight_df.to_string(index=False):<15}')


    # 순위 순으로 정렬
    results_df = results_df.sort_values('순위')
    return results_df, weights

# 5. 결과 시각화
def visualize_results(results_df, weights, criteria, district_col):
    """
    1. 바 차트 - 각 지표의 중요도
    2. Top 5 순위 차트 - 상위 5개 자치구
    3. 레이더 차트 - 1위 자치구의 상세 분석
    4. 전체 점수 분포 - 모든 자치구의 점수 분포
    
    Parameters (입력값):
    - results_df: 분석 결과 데이터프레임
    - weights: 가중치 배열
    - criteria: 평가 기준 리스트
    - district_col: 지역명 컬럼명
    - top_n: 상위 몇 개를 표시할지
    """
    # 전체 figure 생성 (2행 2열 레이아웃)
    fig = plt.figure(figsize=(12, 8))
    
    # 1. 가중치 바 차트
    ax1 = plt.subplot(2, 2, 1)
    weight_df = pd.DataFrame({'기준': criteria, '가중치': weights * 100})
    bars = ax1.barh(weight_df['기준'], weight_df['가중치'], color='steelblue')
    ax1.set_xlabel('가중치 (%)', fontsize=9)
    ax1.set_title('엔트로피 가중치 분석 결과', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontsize=8)

    # 2. Top 5 자치구 순위
    ax2 = plt.subplot(2, 2, 2)
    top_districts = results_df.head(5)
    colors = ['gold', 'silver', 'lightpink', 'lightblue', 'lightgreen']
    bars = ax2.barh(range(5), top_districts['종합점수'].values, color=colors)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(top_districts[district_col].values)], fontsize=9)
    ax2.set_xlabel('종합 점수', fontsize=9)
    ax2.set_title(f'살기 좋은 자치구 Top 5', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center', fontsize=8)

    # 3. 레이더 차트 (글자 겹침 해결을 위한 패딩 추가)
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    top1 = results_df.iloc[0]
    
    normalized_values = []
    for criterion in criteria:
        val = top1[criterion]
        max_val = results_df[criterion].max()
        min_val = results_df[criterion].min()
        range_val = max_val - min_val
        normalized_values.append((val - min_val) / range_val if range_val != 0 else 0.5)
    
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    normalized_values += normalized_values[:1]
    angles += angles[:1]
    
    ax3.plot(angles, normalized_values, 'o-', linewidth=2, color='darkblue')
    ax3.fill(angles, normalized_values, alpha=0.25, color='skyblue')
    ax3.set_xticks(angles[:-1])
    # labelpad를 주어 원본 데이터와 글자 사이의 간격 확보
    ax3.set_xticklabels(criteria, fontsize=9)
    ax3.tick_params(axis='x', pad=15) 
    ax3.set_ylim(0, 1.1) # 글자가 그래프 안으로 들어오지 않게 살짝 여유 부여
    ax3.set_title(f'1위: {top1[district_col]} 상세 분석', fontsize=10, fontweight='bold', pad=30)
    
    # 4. 전체 순위 점수 분포 (텍스트 위치 조정)
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(range(len(results_df)), results_df['종합점수'].values, 
               c=results_df['순위'].values, cmap='RdYlGn_r', s=100, alpha=0.5)
    ax4.set_xlabel('순위', fontsize=9)
    ax4.set_ylabel('종합 점수', fontsize=9)
    ax4.set_title('전체 자치구 점수 분포', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Top 5 텍스트가 서로 겹치지 않도록 번갈아가며 위치 조정
    for i in range(min(5, len(results_df))):
        y_offset = 12 if i % 2 == 0 else -18 # 위아래로 지그재그 배치
        ax4.annotate(results_df.iloc[i][district_col], 
                    (i, results_df.iloc[i]['종합점수']),
                    xytext=(0, y_offset), textcoords='offset points', 
                    ha='center', fontsize=8, fontweight='bold')

    # 자동 레이아웃 설정 (항목 간 간격 확보)
    plt.tight_layout(pad=3.0) 
    plt.savefig('mcdm_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. 메인 실행 함수
def main():
    """
    프로그램의 메인 실행 함수
    모든 분석 과정을 순차적으로 실행
    
    실행 순서:
    1. 데이터 로드
    2. MCDM 분석 수행
    3. 결과 출력
    4. 결과 저장 (엑셀)
    5. 시각화
    """
    # 1. 엑셀 파일 경로
    file_path = r"C:\Users\82109\Desktop\문서.xlsx"
    
    # 2. 지역명이 들어있는 컬럼명
    district_col = '자치구별'
    
    # 3. 평가 기준 컬럼명 리스트
    criteria = [
        '1인 가구',
        '원룸 및 오피스텔',
        '평당 월세 평균',
        '약국',
        '병원',
        '검거율',
        '대규모점포',
        '지하철'        
    ]
    
    # 4. 각 기준이 긍정적(True) 또는 부정적(False)인지 설정
    is_positive = {
        '1인 가구': True,          # 많을수록 좋음
        '원룸 및 오피스텔': True,   # 많을수록 좋음
        '평당 월세 평균': False,         # 낮을수록 좋음
        '약국': True,                # 많을수록 좋음
        '병원': True,              # 많을수록 좋음
        '검거율': True,             # 높을수록 좋음
        '대규모점포': True,         # 많을수록 좋음
        '지하철': True             # 많을수록 좋음
    }

    # 데이터 로드
    df = load_data(file_path)
    
    # MCDM 분석 수행
    results_df, weights = perform_mcdm_analysis(df, district_col, criteria, is_positive)
    
    # 전체 순위 저장
    results_df.to_excel('분석결과_전체순위.xlsx', index=False)

    # 결과 시각화
    visualize_results(results_df, weights, criteria, district_col)
    return results_df, weights


if __name__ == "__main__":
    results, weights = main()
