import { useState, useEffect } from 'react';
import { Calendar, CheckCircle } from 'lucide-react';

// 생산계획 데이터 도면(규격)
interface PlanData {
  id: number;
  week: string;
  client: string;
  product: string;
  plannedQty: number;
  status: string;
}

function Planning() {
  const [plans, setPlans] = useState<PlanData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // 화면이 켜지면 실행되는 스위치
  useEffect(() => {
    // 💡 백엔드 AI 엔진이 완성되기 전이므로, 화면 테스트용 임시 생산 스케줄을 가동합니다.
    const dummyPlans: PlanData[] = [
      { id: 1, week: '2026-09 1주차', client: 'NSK', product: '단조품 A', plannedQty: 400, status: '계획 확정' },
      { id: 2, week: '2026-09 1주차', client: 'SKC', product: '가공품 B', plannedQty: 600, status: '긴급 편성' },
      { id: 3, week: '2026-09 2주차', client: '일진', product: '단조품 C', plannedQty: 250, status: '계획 확정' },
      { id: 4, week: '2026-09 2주차', client: 'SKF', product: '가공품 A', plannedQty: 800, status: '검토 중' },
    ];

    setTimeout(() => {
      setPlans(dummyPlans);
      setIsLoading(false);
    }, 800); // 실제 통신처럼 0.8초 딜레이를 줍니다.
  }, []);

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Calendar size={24} color="#10b981" />
          AI 자동 생산계획 (주간/월간)
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          수요예측 데이터를 기반으로 자동 산출된 최적의 생산 스케줄입니다.
        </p>
      </div>

      {isLoading ? (
        <div style={{ padding: '50px', textAlign: 'center', color: '#64748b' }}>생산 스케줄 시뮬레이션 중...</div>
      ) : (
        <div style={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={thStyle}>생산 주차</th>
                <th style={thStyle}>고객사</th>
                <th style={thStyle}>제품명</th>
                <th style={thStyle}>목표 생산량</th>
                <th style={thStyle}>상태</th>
              </tr>
            </thead>
            <tbody>
              {plans.map((plan) => (
                <tr key={plan.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ ...tdStyle, fontWeight: 'bold', color: '#475569' }}>{plan.week}</td>
                  <td style={tdStyle}>{plan.client}</td>
                  <td style={tdStyle}>{plan.product}</td>
                  <td style={{ ...tdStyle, color: '#10b981', fontWeight: 'bold' }}>{plan.plannedQty.toLocaleString()} 개</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold', display: 'inline-flex', alignItems: 'center', gap: '4px',
                      backgroundColor: plan.status === '긴급 편성' ? '#fef08a' : plan.status === '검토 중' ? '#e2e8f0' : '#dcfce7',
                      color: plan.status === '긴급 편성' ? '#854d0e' : plan.status === '검토 중' ? '#475569' : '#166534'
                    }}>
                      {plan.status === '계획 확정' && <CheckCircle size={14} />}
                      {plan.status}
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

export default Planning;