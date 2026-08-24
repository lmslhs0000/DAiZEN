import { useState } from 'react';
import { Search, TrendingUp, Loader2 } from 'lucide-react';
// 💡 [수정 1] 실제 작동하는 함수(부품)는 일반 import로 가져옵니다.
import { fetchAiForecast } from '../api/forecastApi';
// 💡 [수정 2] 규격(도면)을 의미하는 타입들은 'import type'으로 명확히 분리해서 가져옵니다.
import type { ForecastRequestParams, ForecastResultData } from '../api/forecastApi';

function Forecast() {
  const [client, setClient] = useState('NSK');
  const [productType, setProductType] = useState('단조품');
  const [targetMonth, setTargetMonth] = useState('2026-06');
  const [isLoading, setIsLoading] = useState(false);
  const [forecastData, setForecastData] = useState<ForecastResultData[] | null>(null);

  const handleSearch = async () => {
    setIsLoading(true); 
    
    try {
      const requestParams: ForecastRequestParams = {
        client: client,
        productType: productType,
        targetMonth: targetMonth,
      };

      const result = await fetchAiForecast(requestParams);
      setForecastData(result); 

    } catch (error) {
      // 💡 [수정 3] 가져온 error 변수를 console.error에 실제로 사용하여 ESLint 경고를 해제합니다.
      console.error('API 통신 에러 상세 내용:', error);
      alert('백엔드 서버와 연결할 수 없습니다. 콘솔 창(F12)을 확인해 주세요.');
    } finally {
      setIsLoading(false); 
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={24} color="#1e3a8a" />
          고객/제품별 AI 수요예측 조회
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          백엔드 AI 모델과 연동하여 다가오는 월의 수요를 예측하고 안전재고를 산출합니다.
        </p>
      </div>

      <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', gap: '20px', alignItems: 'flex-end' }}>
        
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>고객사 선택</label>
          <select value={client} onChange={(e) => setClient(e.target.value)} style={inputStyle}>
            <option value="NSK">NSK</option>
            <option value="SKC">SKC</option>
            <option value="일진">일진</option>
            <option value="SKF">SKF</option>
          </select>
        </div>

        <div style={{ flex: 1 }}>
          <label style={labelStyle}>제품군</label>
          <select value={productType} onChange={(e) => setProductType(e.target.value)} style={inputStyle}>
            <option value="단조품">단조품</option>
            <option value="가공품">가공품</option>
          </select>
        </div>

        <div style={{ flex: 1 }}>
          <label style={labelStyle}>예측 기준 월</label>
          <input type="month" value={targetMonth} onChange={(e) => setTargetMonth(e.target.value)} style={inputStyle} />
        </div>

        <button 
          onClick={handleSearch} 
          disabled={isLoading} 
          style={{ ...buttonStyle, opacity: isLoading ? 0.7 : 1 }}
        >
          {isLoading ? (
            <><Loader2 size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} /> 분석 중...</>
          ) : (
            <><Search size={18} /> AI 수요예측 실행</>
          )}
        </button>
      </div>

      <div style={{ marginTop: '25px' }}>
        {forecastData ? (
          <div style={{ padding: '20px', backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#166534' }}>AI 분석이 완료되었습니다.</h3>
            <pre style={{ fontSize: '14px', color: '#15803d' }}>
              {JSON.stringify(forecastData, null, 2)}
            </pre>
          </div>
        ) : (
          <div style={{ padding: '40px', textAlign: 'center', backgroundColor: '#f8fafc', border: '1px dashed #cbd5e1', borderRadius: '8px', color: '#64748b' }}>
            조건을 선택하고 'AI 수요예측 실행' 버튼을 눌러 서버에 데이터를 요청하세요.
          </div>
        )}
      </div>

    </div>
  );
}

const labelStyle = { display: 'block', fontSize: '13px', fontWeight: 'bold', color: '#475569', marginBottom: '8px' };
const inputStyle = { width: '100%', padding: '10px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '14px', outline: 'none' };
const buttonStyle = { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#1e3a8a', color: 'white', padding: '10px 20px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '14px', fontWeight: 'bold', height: '40px' };

export default Forecast;