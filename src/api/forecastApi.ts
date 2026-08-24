// src/api/forecastApi.ts
import { apiClient } from './client';

// 1. 프론트엔드가 백엔드(AI 서버)로 넘겨줄 검색 조건의 '규격'을 정의합니다.
// (NSK, SKC 등 사용자가 선택한 고객사명과 기준 월 정보가 담깁니다.)
export interface ForecastRequestParams {
  client: string;
  productType: string;
  targetMonth: string;
}

// 2. 백엔드 AI 모델이 연산을 마치고 프론트엔드로 돌려줄 데이터의 '결과 규격'입니다.
export interface ForecastResultData {
  month: string;
  예측수요량: number;
  생산계획량: number;
  안전재고: number;
}

// 3. 실제 통신 스위치를 켜는 함수입니다. 화면에서 '검색' 버튼을 누르면 이 함수가 작동합니다.
export const fetchAiForecast = async (params: ForecastRequestParams) => {
  try {
    // apiClient를 통해 백엔드의 '/v1/predict' 주소로 파라미터(고객사 조건)를 보냅니다.
    const response = await apiClient.get<ForecastResultData[]>('/v1/predict', {
      params: params,
    });
    
    // 서버가 데이터를 무사히 돌려주면, 결과값(data)만 뽑아서 화면으로 전달합니다.
    return response.data;
    
  } catch (error) {
    console.error('AI 수요예측 서버 통신 에러:', error);
    throw error; // 화면 쪽으로 에러가 났음을 알립니다.
  }
};