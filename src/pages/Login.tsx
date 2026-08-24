import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, User, Factory } from 'lucide-react';

function Login() {
  // 1. 사용자가 입력한 아이디와 비밀번호를 기억할 상태(State) 공간입니다.
  const [id, setId] = useState('');
  const [pw, setPw] = useState('');
  
  // 2. 화면 이동을 담당하는 네비게이션 모터 역할을 합니다.
  const navigate = useNavigate();

  // 3. 로그인 버튼을 눌렀을 때 실행되는 검증 로직입니다.
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault(); // 폼 제출 시 화면이 새로고침되는 것을 막습니다.

    // 입력된 값이 지정된 출입증(DAIZEN / 0909)과 일치하는지 확인합니다.
    if (id === 'DAIZEN' && pw === '0909') {
      // 일치하면 브라우저의 내장 메모리(localStorage)에 '로그인 성공' 도장을 찍습니다.
      localStorage.setItem('isLoggedIn', 'true');
      // 메인 대시보드 화면('/')으로 즉시 이동시킵니다.
      navigate('/'); 
    } else {
      alert('아이디 또는 비밀번호가 일치하지 않습니다.');
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', backgroundColor: '#0f172a', alignItems: 'center', justifyContent: 'center' }}>
      
      {/* 로그인 박스 컨테이너 */}
      <div style={{ backgroundColor: 'white', padding: '40px', borderRadius: '12px', width: '400px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        
        {/* 로고 및 타이틀 영역 */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '10px' }}>
            <div style={{ backgroundColor: '#1e3a8a', padding: '15px', borderRadius: '50%' }}>
              <Factory size={32} color="white" />
            </div>
          </div>
          <h2 style={{ margin: '0 0 5px 0', color: '#1e3a8a', fontSize: '22px' }}>원강산업 MES</h2>
          <p style={{ margin: 0, color: '#64748b', fontSize: '14px' }}>AI 스마트팩토리 통합 관리체계</p>
        </div>

        {/* 입력 폼 영역 */}
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          
          <div>
            <label style={labelStyle}>아이디</label>
            <div style={inputWrapperStyle}>
              <User size={18} color="#94a3b8" />
              <input 
                type="text" 
                placeholder="아이디를 입력하세요" 
                value={id} 
                onChange={(e) => setId(e.target.value)} 
                style={inputStyle} 
              />
            </div>
          </div>

          <div>
            <label style={labelStyle}>비밀번호</label>
            <div style={inputWrapperStyle}>
              <Lock size={18} color="#94a3b8" />
              <input 
                type="password" 
                placeholder="비밀번호를 입력하세요" 
                value={pw} 
                onChange={(e) => setPw(e.target.value)} 
                style={inputStyle} 
              />
            </div>
          </div>

          <button type="submit" style={buttonStyle}>
            시스템 접속하기
          </button>
          
        </form>
      </div>
    </div>
  );
}

// UI 디자인을 위한 스타일 객체
const labelStyle = { display: 'block', fontSize: '13px', fontWeight: 'bold', color: '#475569', marginBottom: '8px' };
const inputWrapperStyle = { display: 'flex', alignItems: 'center', gap: '10px', border: '1px solid #cbd5e1', padding: '10px 15px', borderRadius: '6px' };
const inputStyle = { border: 'none', outline: 'none', width: '100%', fontSize: '14px' };
const buttonStyle = { marginTop: '10px', backgroundColor: '#1e3a8a', color: 'white', border: 'none', padding: '14px', borderRadius: '6px', fontSize: '15px', fontWeight: 'bold', cursor: 'pointer' };

export default Login;