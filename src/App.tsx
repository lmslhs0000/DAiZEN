import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
// 우리가 깎아둔 화면 부품들을 모두 수입(Import)해옵니다.
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Forecast from './pages/Forecast';
import Planning from './pages/Planning';
import Inventory from './pages/Inventory';
import Login from './pages/Login';

// 🛡️ 보안 게이트웨이: 출입증(로그인) 검사 로직
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isLoggedIn = localStorage.getItem('isLoggedIn');
  if (!isLoggedIn) {
    return <Navigate to="/login" replace />; // 출입증 없으면 로그인 화면으로 추방
  }
  return <>{children}</>; // 출입증 있으면 통과
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 1. 로그인 화면은 게이트 밖에 배치하여 누구나 접근 가능하게 합니다. */}
        <Route path="/login" element={<Login />} />
        
        {/* 2. Layout(메뉴바)과 그 안의 작업 화면들은 보안 게이트 안에 가둡니다. */}
        <Route path="/" element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }>
          {/* Layout 내부의 구멍(<Outlet/>)에 끼워질 알맹이 부품들입니다. */}
          <Route index element={<Dashboard />} /> 
          <Route path="forecast" element={<Forecast />} /> 
          <Route path="planning" element={<Planning />} /> 
          <Route path="inventory" element={<Inventory />} /> 
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;