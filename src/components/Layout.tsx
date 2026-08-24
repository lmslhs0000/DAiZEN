import { Outlet, Link, useNavigate } from 'react-router-dom';
import { LayoutDashboard, TrendingUp, CalendarCheck, PackageSearch, LogOut } from 'lucide-react';

function Layout() {
  // 1. 화면 이동을 담당하는 네비게이션 모터를 장착합니다.
  const navigate = useNavigate();

  // 2. 로그아웃 버튼을 눌렀을 때 실행될 논리입니다.
  const handleLogout = () => {
    // 브라우저 메모리(localStorage)에서 '출입증'을 파기합니다.
    localStorage.removeItem('isLoggedIn');
    // 다시 로그인 화면으로 강제 이동시킵니다.
    navigate('/login');
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', backgroundColor: '#f1f5f9' }}>
      
      {/* 좌측 사이드바 (고정) */}
      <nav style={{ width: '250px', backgroundColor: '#0f172a', color: 'white', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid #1e293b' }}>
          <h1 style={{ fontSize: '18px', fontWeight: 'bold', color: '#38bdf8' }}>원강산업 MES</h1>
          <p style={{ fontSize: '12px', color: '#94a3b8' }}>AI 스마트팩토리 시스템</p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', padding: '15px 10px', gap: '5px' }}>
          <Link to="/" style={menuStyle}>
            <LayoutDashboard size={20} /> 종합 대시보드
          </Link>
          <Link to="/forecast" style={menuStyle}>
            <TrendingUp size={20} /> AI 수요예측 조회
          </Link>
          <Link to="/planning" style={menuStyle}>
            <CalendarCheck size={20} /> 자동 생산계획 수립
          </Link>
          <Link to="/inventory" style={menuStyle}>
            <PackageSearch size={20} /> 적정재고 시뮬레이션
          </Link>
        </div>
      </nav>

      {/* 우측 메인 영역 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        
        {/* 상단 헤더 (고정) */}
        <header style={{ height: '60px', backgroundColor: 'white', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#475569' }}>
            수주-계획-생산-재고 통합 관리체계
          </div>
          
          {/* 💡 사용자 정보 및 로그아웃 버튼 영역 */}
          <div style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '15px' }}>
            <span style={{ backgroundColor: '#dcfce7', color: '#166534', padding: '4px 10px', borderRadius: '15px', fontWeight: 'bold' }}>
              🟢 대명 창원/마산 연동 중
            </span>
            <span style={{ color: '#64748b' }}><strong>DAIZEN</strong> 관리자님</span>
            
            {/* 3. 로그아웃 버튼 조립 */}
            <button onClick={handleLogout} style={logoutButtonStyle}>
              <LogOut size={16} /> 시스템 종료
            </button>
          </div>
        </header>

        {/* 실제 콘텐츠가 바뀌는 본문 영역 */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
          <Outlet />
        </main>

      </div>
    </div>
  );
}

// 메뉴 버튼 스타일
const menuStyle = { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 15px', color: '#cbd5e1', textDecoration: 'none', fontSize: '14px', borderRadius: '6px', transition: 'background 0.2s' };

// 로그아웃 버튼 스타일
const logoutButtonStyle = { display: 'flex', alignItems: 'center', gap: '5px', backgroundColor: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' };

export default Layout;