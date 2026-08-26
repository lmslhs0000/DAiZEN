import { useState, useEffect } from 'react';
import { Package, AlertTriangle, CheckCircle } from 'lucide-react';

// 1. 재고 데이터 도면(규격)
interface InventoryData {
  id: number;
  client: string;
  product: string;
  currentStock: number;
  safetyStock: number;
  status: string;
}

function Inventory() {
  const [inventory, setInventory] = useState<InventoryData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // 2. 화면이 켜지면 실행되는 스위치
  useEffect(() => {
    // 백엔드 AI 엔진이 완성되기 전, 화면 테스트용 임시 재고 현황을 가동합니다.
    const dummyInventory: InventoryData[] = [
      { id: 1, client: 'NSK', product: '단조품 A', currentStock: 250, safetyStock: 300, status: '위험 (재고부족)' },
      { id: 2, client: 'SKC', product: '가공품 B', currentStock: 800, safetyStock: 500, status: '안정' },
      { id: 3, client: '일진', product: '단조품 C', currentStock: 120, safetyStock: 150, status: '위험 (재고부족)' },
      { id: 4, client: 'SKF', product: '가공품 A', currentStock: 650, safetyStock: 600, status: '안정' },
    ];

    setTimeout(() => {
      setInventory(dummyInventory);
      setIsLoading(false);
    }, 800);
  }, []);

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Package size={24} color="#f59e0b" />
          적정재고 및 안전재고 시뮬레이션
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          AI 수요예측 및 생산계획과 연동된 실시간 재고 현황 및 최적화 모니터링입니다.
        </p>
      </div>

      {isLoading ? (
        <div style={{ padding: '50px', textAlign: 'center', color: '#64748b' }}>재고 현황 데이터를 불러오는 중...</div>
      ) : (
        <div style={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={thStyle}>고객사</th>
                <th style={thStyle}>제품명</th>
                <th style={thStyle}>현재 재고량</th>
                <th style={thStyle}>안전 재고 기준</th>
                <th style={thStyle}>재고 건전성 상태</th>
              </tr>
            </thead>
            <tbody>
              {inventory.map((item) => (
                <tr key={item.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ ...tdStyle, fontWeight: 'bold' }}>{item.client}</td>
                  <td style={tdStyle}>{item.product}</td>
                  {/* 현재 재고가 안전 재고보다 적으면 빨간색으로 경고 표시를 합니다. */}
                  <td style={{ ...tdStyle, fontWeight: 'bold', color: item.currentStock < item.safetyStock ? '#ef4444' : '#0f172a' }}>
                    {item.currentStock.toLocaleString()} 개
                  </td>
                  <td style={{ ...tdStyle, color: '#64748b' }}>{item.safetyStock.toLocaleString()} 개</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold', display: 'inline-flex', alignItems: 'center', gap: '4px',
                      backgroundColor: item.status.includes('위험') ? '#fee2e2' : '#dcfce7',
                      color: item.status.includes('위험') ? '#991b1b' : '#166534'
                    }}>
                      {item.status.includes('위험') ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
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

export default Inventory;