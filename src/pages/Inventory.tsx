import { useState } from 'react';
import { PackageSearch, Sliders, ArrowRight, Activity } from 'lucide-react';

function Inventory() {
  // 사용자가 슬라이더로 조절할 수 있는 시뮬레이션 변수(상태)들입니다.
  const [safetyMargin, setSafetyMargin] = useState(15); // 안전재고 여유율 (%)
  const [demandVolatility, setDemandVolatility] = useState(10); // 수요 변동폭 (%)

  // 가상의 기본 재고 데이터 (백엔드에서 가져올 기준값)
  const baseStock = 2000;
  
  // 시뮬레이션 공식: 기본 재고 + (여유율 적용) + (수요 변동에 따른 추가 확보)
  const calculatedOptimalStock = Math.round(baseStock * (1 + safetyMargin / 100) * (1 + demandVolatility / 100));

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      
      {/* 헤더 영역 */}
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ color: '#0f172a', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <PackageSearch size={24} color="#1e3a8a" />
          적정재고 및 안전재고 시뮬레이션
        </h2>
        <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
          대명 창원/마산 공장의 현재 재고를 기준으로, 공급망 변동성에 대비한 최적의 안전재고를 산출합니다.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
        
        {/* 1. 좌측: 시뮬레이션 조건 설정 패널 */}
        <div style={{ flex: 1, backgroundColor: 'white', padding: '25px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px', color: '#334155' }}>
            <Sliders size={18} /> 시뮬레이션 파라미터 제어
          </h3>
          
          <div style={{ marginTop: '20px' }}>
            <label style={labelStyle}>
              안전재고 여유율 (Safety Margin): <strong>{safetyMargin}%</strong>
            </label>
            <input 
              type="range" 
              min="0" max="50" 
              value={safetyMargin} 
              onChange={(e) => setSafetyMargin(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
            <p style={helpTextStyle}>긴급 발주나 납기 지연을 대비하여 추가로 확보할 재고의 비율입니다.</p>
          </div>

          <div style={{ marginTop: '25px' }}>
            <label style={labelStyle}>
              시장 수요 변동폭 (Volatility): <strong>{demandVolatility}%</strong>
            </label>
            <input 
              type="range" 
              min="0" max="30" 
              value={demandVolatility} 
              onChange={(e) => setDemandVolatility(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
            <p style={helpTextStyle}>고객사(NSK, SKC 등)의 갑작스러운 주문량 증가 예상 수치입니다.</p>
          </div>
        </div>

        {/* 화살표 아이콘 */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', paddingTop: '100px' }}>
          <ArrowRight size={32} color="#94a3b8" />
        </div>

        {/* 2. 우측: 시뮬레이션 결과 표출 패널 */}
        <div style={{ flex: 1, backgroundColor: '#1e3a8a', padding: '25px', borderRadius: '8px', color: 'white' }}>
          <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px', color: '#93c5fd' }}>
            <Activity size={18} /> AI 추천 적정재고 산출 결과
          </h3>
          
          <div style={{ marginTop: '30px', textAlign: 'center' }}>
            <p style={{ color: '#cbd5e1', margin: '0 0 10px 0' }}>선택하신 부품의 현재 기준 재고: {baseStock.toLocaleString()} 개</p>
            <div style={{ fontSize: '48px', fontWeight: '900', margin: '10px 0' }}>
              {calculatedOptimalStock.toLocaleString()} <span style={{ fontSize: '20px', fontWeight: 'normal', color: '#93c5fd' }}>개</span>
            </div>
            
            <div style={{ marginTop: '20px', backgroundColor: 'rgba(255,255,255,0.1)', padding: '15px', borderRadius: '6px', fontSize: '13px' }}>
              현재 설정된 변동성({demandVolatility}%)과 여유율({safetyMargin}%)을 고려할 때, <br/>
              결품 방지와 재고자산 최소화를 동시에 달성할 수 있는 최적 수량입니다.
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

const labelStyle = { display: 'block', fontSize: '14px', fontWeight: 'bold', color: '#475569', marginBottom: '10px' };
const helpTextStyle = { fontSize: '12px', color: '#94a3b8', marginTop: '5px' };

export default Inventory;