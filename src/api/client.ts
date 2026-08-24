// src/api/client.ts
import axios from 'axios';

// 메인 통신 객체 생성: 프론트엔드의 모든 데이터 요청은 이 규격을 따릅니다.
export const apiClient = axios.create({
  // 백엔드 팀원이 서버를 완성하면 이 주소를 실제 서버 주소로 변경합니다.
  // 현재는 협업 개발 중 가장 흔히 쓰는 로컬 테스트 주소를 미리 적어둡니다.
  baseURL: 'http://localhost:8000/api', 
  headers: {
    'Content-Type': 'application/json',
  },
  // 통신을 요청하고 10초(10000ms) 동안 응답이 없으면 에러를 발생시킵니다.
  // (시스템이 무한 대기 상태에 빠지는 것을 방지하는 안전장치)
  timeout: 10000, 
});