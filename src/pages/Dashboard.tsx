import { 
    ResponsiveContainer, 
    ComposedChart, 
    Line, 
    Bar, 
    XAxis, 
    YAxis, 
    CartesianGrid, 
    Tooltip, 
    Legend 
  } from 'recharts';
  import { TrendingUp, Package, Activity } from 'lucide-react';
  
  // 1. 가상의 AI 수요예측 데이터 (추후 Python 백엔드 AI 모델에서 받아올 데이터 형식입니다)
  const aiForecastData = [
    { month: '1월', 예측수요량: 4200, 생산계획량: 4500, 안전재고: 800 },
    { month: '2월', 예측수요량: 3800, 생산계획량: 4000, 안전재고: 1000 },
    { month: '3월', 예측수요량: 5500, 생산계획량: 5200, 안전재고: 500 }, // 수요 급증 구간
    { month: '4월', 예측수요량: 4800, 생산계획량: 5000, 안전재고: 700 },
    { month: '5월', 예측수요량: 4100, 생산계획량: 4300, 안전재고: 800 },
    { month: '6월', 예측수요량: 3900, 생산계획량: 4000, 안전재고: 900 },
  ];
  
  function Dashboard() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* 2. 상단 타이틀 영역 */}
        <div>
          <h2 style={{ color: '#0f172a', margin: '0 0 5px 0' }}>📈 종합 수요예측 및 생산 관제</h2>
          <p style={{ color: '#64748b', margin: 0, fontSize: '14px' }}>
            AI 모델 기반 고객사(NSK, SKC 등) 발주 예측 및 대명 창원/마산 공장 재고 연동 현황
          </p>
        </div>
  
        {/* 3. KPI 요약 카드 영역 (프로젝트 기대효과 지표 표출) */}
        <div style={{ display: 'flex', gap: '20px' }}>
          <KpiCard 
            title="AI 예측 정확도" 
            value="94.2%" 
            trend="▲ 2.1% (전월대비)" 
            icon={<TrendingUp size={24} color="#2563eb" />} 
            trendColor="#166534"
          />
          <KpiCard 
            title="평균 재고 회전일수" 
            value="15일" 
            trend="▼ 3일 감소 (재고 최적화)" 
            icon={<Package size={24} color="#16a34a" />} 
            trendColor="#166534"
          />
          <KpiCard 
            title="실시간 공장 가동률" 
            value="88.5%" 
            trend="적정 부하 유지중" 
            icon={<Activity size={24} color="#dc2626" />} 
            trendColor="#64748b"
          />
        </div>
  
        {/* 4. Recharts를 활용한 AI 시뮬레이션 차트 영역 */}
        <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <h3 style={{ margin: '0 0 20px 0', fontSize: '16px', color: '#1e3a8a' }}>
            상반기 수요예측 vs 생산계획 시뮬레이션 (단위: 개)
          </h3>
          
          {/* 화면 크기에 맞춰 차트 크기를 자동 조절해주는 컨테이너 */}
          <div style={{ width: '100%', height: '400px' }}>
            <ResponsiveContainer>
              <ComposedChart data={aiForecastData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" />
                <XAxis dataKey="month" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip cursor={{ fill: '#f8fafc' }} />
                <Legend />
                
                {/* 막대그래프 (수요예측량 및 생산계획량) */}
                <Bar dataKey="예측수요량" fill="#94a3b8" barSize={30} radius={[4, 4, 0, 0]} />
                <Bar dataKey="생산계획량" fill="#2563eb" barSize={30} radius={[4, 4, 0, 0]} />
                
                {/* 꺾은선그래프 (안전재고 추이 자동 산출) */}
                <Line type="monotone" dataKey="안전재고" stroke="#dc2626" strokeWidth={3} dot={{ r: 4 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
  
      </div>
    );
  }
  
  // 💡 미니 부품(컴포넌트): KPI 카드를 반복해서 쓰기 위해 분리해 둔 블록입니다.
  function KpiCard({ title, value, trend, icon, trendColor }: { title: string, value: string, trend: string, icon: React.ReactNode, trendColor: string }) {
    return (
      <div style={{ flex: 1, backgroundColor: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', gap: '15px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
        <div style={{ backgroundColor: '#f1f5f9', padding: '15px', borderRadius: '50%' }}>
          {icon}
        </div>
        <div>
          <div style={{ fontSize: '13px', color: '#64748b', fontWeight: 'bold' }}>{title}</div>
          <div style={{ fontSize: '24px', fontWeight: '900', color: '#0f172a', margin: '4px 0' }}>{value}</div>
          <div style={{ fontSize: '12px', color: trendColor, fontWeight: 'bold' }}>{trend}</div>
        </div>
      </div>
    );
  }
  
  export default Dashboard;