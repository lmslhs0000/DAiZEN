import { useState, useEffect } from 'react';
import { TrendingUp, AlertTriangle } from 'lucide-react';
// 1. 대시보드와 똑같이 '변경된 창고 위치'와 '변경된 부품명'을 사용합니다.
import { fetchAiForecast } from '../api/forecastApi';

// 2. 표(Table)에 들어갈 데이터의 규격(도면)을 이 파일 안에서 자체적으로 정의합니다.
interface TableData {
  id: number;
  client: string;
  product: string;
  predictedDemand: number;
  safetyStock: number;
  status: string;
}

function Forecast() {
  const [data, setData] = useState<TableData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // 3. 화면이 켜지면 데이터를 즉시 불러오는 자동 스위치
  useEffect(() => {
    const loadData = async () => {
      try {
        // 서버에 통신망이 잘 연결되어 있는지 찔러봅니다.
        await fetchAiForecast({ client: 'NSK', productType: '단조품', targetMonth: '2026-09' });
        
        // 통신이 성공하든 실패하든, 아직 백엔드에 '전체 목록' API가 없으므로 
        // 화면 디자인을 확인하기 위해 비상용 가짜 데이터를 표에 채워 넣습니다.
        setData(dummyList);
      } catch (error) {
        console.error("데이터 로딩 중 에러 발생:", error);
        setData(dummyList); // 에러가 나도 표가 보이게 세팅
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  // 비상용 가짜 데이터 부품
  const dummyList: TableData[] = [
    { id: 1, client: 'NSK', product: '단조품 A', predictedDemand: 1500, safetyStock: 300, status: '안정' },
    { id: 2, client: 'SKC', product: '가공품 B', predictedDemand: 2200, safetyStock: 500, status: '위험 (재고부족)' },
    { id: 3, client: '일진', product: '단조품 C', predictedDemand: 800, safetyStock: 150, status: '안정' },
    { id: 4, client: 'SKF', product: '가공품 A', predictedDemand: 3100, safetyStock: 600, status: '주의 (수요급증)' },
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={24} color="#1e3a8a" />
          AI 기반 수요예측 상세 현황
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          고객사별 단조 및 가공 제품의 다음 달 예측 수요량과 상태를 모니터링합니다.
        </p>
      </div>

      {isLoading ? (
        <div style={{ padding: '50px', textAlign: 'center', color: '#64748b' }}>데이터를 분석 중입니다...</div>
      ) : (
        <div style={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={thStyle}>고객사</th>
                <th style={thStyle}>제품명</th>
                <th style={thStyle}>예측 수요량 (AI 산출)</th>
                <th style={thStyle}>안전 재고 기준</th>
                <th style={thStyle}>수급 상태</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item) => (
                <tr key={item.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ ...tdStyle, fontWeight: 'bold' }}>{item.client}</td>
                  <td style={tdStyle}>{item.product}</td>
                  <td style={{ ...tdStyle, color: '#1e3a8a', fontWeight: 'bold' }}>{item.predictedDemand.toLocaleString()} 개</td>
                  <td style={{ ...tdStyle, color: '#64748b' }}>{item.safetyStock.toLocaleString()} 개</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold', display: 'inline-flex', alignItems: 'center', gap: '4px',
                      backgroundColor: item.status.includes('안정') ? '#dcfce7' : '#fee2e2',
                      color: item.status.includes('안정') ? '#166534' : '#991b1b'
                    }}>
                      {item.status.includes('위험') && <AlertTriangle size={14} />}
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const thStyle = { padding: '15px', fontSize: '14px', color: '#475569' };
const tdStyle = { padding: '15px', fontSize: '14px', color: '#0f172a' };

export default Forecast;