import { CalendarCheck, CheckCircle, AlertTriangle } from 'lucide-react';

function Planning() {
  // 1. 가상의 AI 산출 생산계획 데이터 (배열 형태)
  // 백엔드에서 이와 같은 형태로 데이터를 넘겨주게 됩니다.
  const productionPlans = [
    { id: 1, week: '2026-06 1주차', client: 'NSK', product: '단조품 A', demand: 1200, plan: 1500, stock: 300, status: '안정' },
    { id: 2, week: '2026-06 2주차', client: 'NSK', product: '단조품 A', demand: 1500, plan: 1500, stock: 300, status: '안정' },
    { id: 3, week: '2026-06 3주차', client: 'NSK', product: '단조품 A', demand: 2800, plan: 2000, stock: -500, status: '부족경고' },
    { id: 4, week: '2026-06 4주차', client: 'SKC', product: '가공품 B', demand: 800, plan: 1000, stock: 200, status: '안정' },
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      
      {/* 헤더 영역 */}
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CalendarCheck size={24} color="#1e3a8a" />
          AI 주간/월간 자동 생산계획
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          수요예측 결과를 바탕으로 도출된 최적의 생산 스케줄 및 적정재고 시뮬레이션 결과입니다.
        </p>
      </div>

      {/* 데이터 그리드(표) 영역 */}
      <div style={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          
          {/* 표의 머리글 (Header) */}
          <thead style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
            <tr>
              <th style={thStyle}>생산 주차</th>
              <th style={thStyle}>고객사 / 제품</th>
              <th style={thStyle}>AI 예측 수요량</th>
              <th style={thStyle}>추천 생산계획량</th>
              <th style={thStyle}>예상 재고(안전재고)</th>
              <th style={thStyle}>상태</th>
            </tr>
          </thead>
          
          {/* 표의 본문 (Body) - map 함수를 이용해 데이터 개수만큼 행을 자동 생성합니다 */}
          <tbody>
            {productionPlans.map((row) => (
              <tr key={row.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                <td style={tdStyle}>{row.week}</td>
                <td style={tdStyle}>
                  <strong>{row.client}</strong> - {row.product}
                </td>
                <td style={tdStyle}>{row.demand.toLocaleString()} 개</td>
                <td style={tdStyle}><strong style={{ color: '#2563eb' }}>{row.plan.toLocaleString()} 개</strong></td>
                <td style={tdStyle}>
                  {row.stock < 0 ? (
                    <span style={{ color: '#dc2626', fontWeight: 'bold' }}>{row.stock} 개 (재고 부족)</span>
                  ) : (
                    <span>{row.stock} 개</span>
                  )}
                </td>
                <td style={tdStyle}>
                  {row.status === '안정' ? (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#16a34a', fontSize: '13px', fontWeight: 'bold' }}>
                      <CheckCircle size={14} /> 안정
                    </span>
                  ) : (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#dc2626', fontSize: '13px', fontWeight: 'bold' }}>
                      <AlertTriangle size={14} /> 부족 경고
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}

// 표 디자인을 위한 스타일 객체
const thStyle = { padding: '15px 20px', fontSize: '13px', color: '#475569', fontWeight: 'bold' };
const tdStyle = { padding: '15px 20px', fontSize: '14px', color: '#0f172a' };

export default Planning;