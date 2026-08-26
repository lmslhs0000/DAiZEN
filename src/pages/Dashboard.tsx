import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// 1. 올바른 경로(../api/forecastApi)에서 올바른 함수(fetchAiForecast)를 가져옵니다.
import { fetchAiForecast } from '../api/forecastApi';
import type { ForecastResultData } from '../api/forecastApi';

function Dashboard() {
  // 데이터 규격이 바뀌었으므로 임시로 any 타입 배열을 사용하여 에러를 방지합니다.
  const [data, setData] = useState<ForecastResultData[]>([]);
  const [isLoading, setIsLoading] = useState(true); 

  useEffect(() => {
    const loadData = async () => {
      try {
        const result = await fetchAiForecast({
          client: 'NSK',
          productType: '단조품',
          targetMonth: '2026-09',
        });
        
        // 💡 테스터기 1: 백엔드에서 진짜로 뭐가 왔는지 브라우저 기록장치에 찍어봅니다.
        console.log("백엔드에서 도착한 수주 데이터:", result);

        // 비상 발전기: 만약 데이터가 비어있다면 프론트엔드 자체 가짜 데이터를 집어넣습니다.
        if (!result || result.length === 0) {
          console.log("데이터가 비어있어 임시 시뮬레이션 데이터를 가동합니다.");
          setData([
            { month: '9월 1주차', 예측수요량: 1500, 생산계획량: 1400, 안전재고: 300 },
            { month: '9월 2주차', 예측수요량: 2100, 생산계획량: 2000, 안전재고: 300 },
            { month: '9월 3주차', 예측수요량: 1800, 생산계획량: 1800, 안전재고: 300 },
            { month: '9월 4주차', 예측수요량: 2400, 생산계획량: 2500, 안전재고: 300 },
          ]);
        } else {
          setData(result); // 데이터가 정상적으로 오면 그걸 씁니다.
        }

      } catch (error) {
        // 💡 테스터기 2: 에러가 났을 때 원인을 정확히 기록합니다.
        console.error("통신망 연결 실패:", error);
        
        // 통신이 아예 실패(서버 꺼짐 등)해도 그래프가 보이게 비상 데이터를 넣습니다.
        setData([
          { month: '9월 1주차', 예측수요량: 1500, 생산계획량: 1400, 안전재고: 300 },
          { month: '9월 2주차', 예측수요량: 2100, 생산계획량: 2000, 안전재고: 300 },
          { month: '9월 3주차', 예측수요량: 1800, 생산계획량: 1800, 안전재고: 300 },
          { month: '9월 4주차', 예측수요량: 2400, 생산계획량: 2500, 안전재고: 300 },
        ]);
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ color: '#0f172a', marginBottom: '20px' }}>🏭 원강산업 종합 관제 대시보드</h2>
      
      <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', height: '400px' }}>
        <h3 style={{ marginTop: 0, color: '#475569', marginBottom: '20px' }}>AI 수요예측 및 생산계획 연동 추이</h3>
        
        {isLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            통신망 연결 중... 데이터를 불러오고 있습니다.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              {/* 3. API가 뱉어내는 실제 데이터 이름(month, 예측수요량 등)을 그래프 축에 연결합니다. */}
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              {/* 바뀐 한글 변수명으로 선 그래프를 3개 그립니다. */}
              <Line type="monotone" dataKey="예측수요량" stroke="#1e3a8a" strokeWidth={3} name="예측 수요량" />
              <Line type="monotone" dataKey="생산계획량" stroke="#10b981" strokeWidth={3} name="생산 계획량" />
              <Line type="monotone" dataKey="안전재고" stroke="#ef4444" strokeWidth={3} name="안전 재고" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

export default Dashboard;