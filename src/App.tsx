import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Forecast from './pages/Forecast';
import Planning from './pages/Planning';
import Inventory from './pages/Inventory';
import Login from './pages/Login'; // 👈 [추가] 방금 만든 로그인 부품 수입

// 💡 보안 게이트웨이 부품: 출입증(로그인 기록)이 없으면 쫓아냅니다.
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  // 브라우저 메모리에 '로그인 성공' 기록이 있는지 확인합니다.
  const isLoggedIn = localStorage.getItem('isLoggedIn');
  
  if (!isLoggedIn) {
    // 기록이 없으면 로그인 페이지로 강제 이동시킵니다.
    return <Navigate to="/login" replace />;
  }
  // 기록이 있으면 원래 보려던 화면(children)을 보여줍니다.
  return <>{children}</>;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 1. 로그인 화면은 누구나 들어올 수 있도록 보안 게이트 밖에 둡니다. */}
        <Route path="/login" element={<Login />} />
        
        {/* 2. 대시보드를 포함한 모든 작업 화면은 <ProtectedRoute> 라는 보안 게이트 안에 가둡니다. */}
        <Route path="/" element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }>
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